import copy  # 객체의 깊은 복사를 위한 모듈
import os  # 운영 체제와의 상호작용을 위한 모듈
import pickle  # 파이썬 객체 직렬화를 위한 모듈
from pathlib import Path  # 파일 경로 처리를 위한 모듈
from functools import partial  # 함수의 인수 고정을 위한 모듈

import numpy as np  # 수치 연산을 위한 라이브러리
import torch  # 파이토치: 딥러닝을 위한 텐서 라이브러리
import torch.nn as nn  # 파이토치의 신경망 모듈
import torch.nn.functional as F  # 파이토치의 함수형 API
from einops import reduce  # 다차원 배열 조작을 위한 라이브러리
from p_tqdm import p_map  # 멀티프로세싱 및 진행 바를 위한 모듈
from pytorch3d.transforms import (axis_angle_to_quaternion,  # 축-각 변환을 쿼터니언으로 변환하는 함수
                                  quaternion_to_axis_angle)  # 쿼터니언을 축-각 변환으로 변환하는 함수
from tqdm import tqdm  # 반복문의 진행 상태를 시각화하는 모듈

from dataset.quaternion import ax_from_6v, quat_slerp  # 데이터셋에서 쿼터니언 관련 유틸리티 함수
from vis import skeleton_render  # 스켈레톤 렌더링 함수

from .utils import extract, make_beta_schedule  # 모듈 내 유틸리티 함수

def identity(t, *args, **kwargs):
    # 입력을 그대로 반환하는 함수
    return t

class EMA:
    # Exponential Moving Average (EMA) 클래스
    def __init__(self, beta):
        # 클래스 초기화 메서드
        super().__init__()
        self.beta = beta  # EMA의 감쇠 계수 설정

    def update_model_average(self, ma_model, current_model):
        # EMA 모델 업데이트 메서드
        # 현재 모델의 파라미터를 EMA 모델의 파라미터로 업데이트
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        # 새로운 값으로 EMA 업데이트
        # 이전 값과 새로운 값을 사용하여 지수 이동 평균을 계산
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new



class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion 클래스는 딥러닝 모델을 사용하여 시간에 따라 점진적으로 데이터를 변형하는 프로세스를 구현합니다.
    """

    def __init__(
        self,
        model,
        horizon,
        repr_dim,
        smpl,
        n_timestep=1000,
        schedule="linear",
        loss_type="l1",
        clip_denoised=True,
        predict_epsilon=True,
        guidance_weight=3,
        use_p2=False,
        cond_drop_prob=0.2,
    ):
        # 클래스 초기화 메서드
        super().__init__()
        self.horizon = horizon  # 예측 시간 길이
        self.transition_dim = repr_dim  # 표현 차원
        self.model = model  # 사용할 모델
        self.ema = EMA(0.9999)  # EMA 클래스 인스턴스
        self.master_model = copy.deepcopy(self.model)  # 모델의 깊은 복사본

        self.cond_drop_prob = cond_drop_prob  # 조건 드롭 확률

        # FK 모듈을 위한 SMPL 인스턴스 생성
        self.smpl = smpl

        # 스케줄에 따른 베타 값 생성
        betas = torch.Tensor(
            make_beta_schedule(schedule=schedule, n_timestep=n_timestep)
        )
        alphas = 1.0 - betas  # 알파 값 계산 (1에서 베타 값을 뺀 값)
        alphas_cumprod = torch.cumprod(alphas, axis=0)  # 알파 누적 곱 계산 (알파 값을 차례로 곱한 값)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])  # 이전 알파 누적 곱 (첫 번째 값을 1로 설정)

        self.n_timestep = int(n_timestep)  # 타임스텝 수 설정
        self.clip_denoised = clip_denoised  # 노이즈 제거 클립 여부
        self.predict_epsilon = predict_epsilon  # 엡실론 예측 여부

        # 베타 값 버퍼에 등록 (모델 학습 시 사용)
        self.register_buffer("betas", betas)
        # 알파 누적 곱 버퍼에 등록
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        # 이전 알파 누적 곱 버퍼에 등록
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.guidance_weight = guidance_weight  # 가이던스 가중치 설정

        # diffusion q(x_t | x_{t-1}) 및 기타 계산을 위한 값들 등록
        # 알파 누적 곱의 제곱근
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        # 1에서 알파 누적 곱을 뺀 값의 제곱근
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        # 1에서 알파 누적 곱을 뺀 값의 로그
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        # 알파 누적 곱의 역수의 제곱근
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        # (알파 누적 곱의 역수 - 1)의 제곱근
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # posterior q(x_{t-1} | x_t, x_0) 계산을 위한 값들 등록
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # posterior_variance 버퍼에 등록
        self.register_buffer("posterior_variance", posterior_variance)

        ## posterior_variance는 확산 체인의 시작 부분에서 0이 되므로 log 계산 클립
        # posterior_variance의 값을 최소 1e-20으로 클램핑하고 로그 값을 버퍼에 등록
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        # posterior_mean_coef1 버퍼에 등록 (베타 값과 이전 알파 누적 곱의 제곱근을 사용하여 계산)
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        # posterior_mean_coef2 버퍼에 등록 (알파 값과 이전 알파 누적 곱을 사용하여 계산)
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # p2 weighting 설정
        self.p2_loss_weight_k = 1
        self.p2_loss_weight_gamma = 0.5 if use_p2 else 0
        # p2_loss_weight 버퍼에 등록 (알파 누적 곱을 사용하여 p2 손실 가중치 계산)
        self.register_buffer(
            "p2_loss_weight",
            (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -self.p2_loss_weight_gamma,
        )

        ## 손실 함수 설정 및 초기화
        # 손실 함수 설정 (loss_type에 따라 l1 또는 l2 손실 함수 선택)
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss

    # ------------------------------------------ sampling ------------------------------------------#


    def predict_start_from_noise(self, x_t, t, noise):
        """
        주어진 노이즈로부터 x_0을 예측하는 함수입니다.
        모델이 epsilon을 예측하면, 노이즈를 이용해 x_0을 계산합니다.
        그렇지 않으면 모델이 직접 x_0을 예측합니다.
        """
        if self.predict_epsilon:
            # 모델이 노이즈를 예측하는 경우
            # sqrt_recip_alphas_cumprod와 sqrt_recipm1_alphas_cumprod를 사용하여 x_t에서 x_0을 계산
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t  # sqrt_recip_alphas_cumprod 값을 추출하여 x_t에 곱함
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise  # sqrt_recipm1_alphas_cumprod 값을 추출하여 noise에 곱한 후 위의 결과에서 뺌
            )
        else:
            # 모델이 x_0을 직접 예측하는 경우
            return noise  # 노이즈 자체를 반환

    def predict_noise_from_start(self, x_t, t, x0):
        """
        주어진 x_0으로부터 노이즈를 예측하는 함수입니다.
        """
        # x_t와 x_0을 이용하여 노이즈를 계산
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /  # sqrt_recip_alphas_cumprod 값을 추출하여 x_t에 곱한 후 x0을 뺌
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)  # sqrt_recipm1_alphas_cumprod 값으로 나눔
        )

    def model_predictions(self, x, cond, t, weight=None, clip_x_start=False):
        """
        모델의 예측을 수행하는 함수입니다.
        조건과 시간 t에서 모델의 출력을 예측하고, 필요한 경우 클리핑합니다.
        """
        weight = weight if weight is not None else self.guidance_weight  # 가이던스 가중치를 설정
        model_output = self.model.guided_forward(x, cond, t, weight)  # 모델의 guided_forward 함수를 사용하여 예측
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity  # 필요한 경우 클리핑 함수 정의
        
        x_start = model_output  # 모델의 출력을 x_start로 설정
        x_start = maybe_clip(x_start)  # x_start를 클리핑
        pred_noise = self.predict_noise_from_start(x, t, x_start)  # x_start로부터 노이즈를 예측

        return pred_noise, x_start  # 예측된 노이즈와 x_start를 반환

    def q_posterior(self, x_start, x_t, t):
        """
        q(x_{t-1} | x_t, x_0)의 posterior mean과 variance를 계산하는 함수입니다.
        """
        # posterior mean 계산
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start  # posterior_mean_coef1을 추출하여 x_start에 곱함
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t  # posterior_mean_coef2를 추출하여 x_t에 곱한 후 위의 결과에 더함
        )
        # posterior variance 계산
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)  # posterior_variance 값을 추출하여 계산
        # posterior log variance 계산
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape  # posterior_log_variance_clipped 값을 추출하여 계산
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped  # 계산된 posterior mean, variance, log variance를 반환

    def p_mean_variance(self, x, cond, t):
        """
        모델의 평균과 분산을 계산하는 함수입니다.
        """
        # 가이던스 클리핑
        if t[0] > 1.0 * self.n_timestep:
            weight = min(self.guidance_weight, 0)  # t가 타임스텝의 1배를 초과하면 weight를 0으로 설정
        elif t[0] < 0.1 * self.n_timestep:
            weight = min(self.guidance_weight, 1)  # t가 타임스텝의 0.1배 미만이면 weight를 1로 설정
        else:
            weight = self.guidance_weight  # 그렇지 않으면 기본 가이던스 가중치를 사용

        # 노이즈로부터 시작점을 예측
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.model.guided_forward(x, cond, t, weight)
        )

        # 필요시 x_recon 클램핑
        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)  # x_recon 값을 -1.0에서 1.0 사이로 클램핑
        else:
            assert RuntimeError()  # clip_denoised가 False인 경우 에러 발생

        # posterior mean과 variance 계산
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_recon  # 계산된 model_mean, posterior_variance, posterior_log_variance, x_recon을 반환

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        """
        현재 시점 t에서 샘플 x를 생성하는 함수입니다.
        노이즈가 추가된 모델의 출력을 반환합니다.
        """
        b, *_, device = *x.shape, x.device  # 입력 x의 배치 크기와 장치(device)를 추출
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, cond=cond, t=t  # 모델의 평균과 로그 분산, 그리고 x_start를 계산
        )
        noise = torch.randn_like(model_mean)  # 모델 평균과 같은 형태의 노이즈를 생성
        # t가 0일 때 노이즈가 추가되지 않도록 마스크 생성
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(noise.shape) - 1))  # 배치 크기를 제외한 나머지 차원은 1로 설정
        )
        # 모델 평균과 노이즈를 결합하여 출력 x_out을 계산
        x_out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_out, x_start  # 샘플링된 x_out과 x_start를 반환

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        """
        주어진 형태(shape)와 조건(cond)에서 샘플링을 반복하는 함수입니다.
        확산 과정의 전체 또는 부분을 시뮬레이션합니다.
        """
        device = self.betas.device  # 모델 파라미터가 있는 장치를 가져옴

        # 기본적으로 전체 타임스케일에 대한 확산을 수행
        start_point = self.n_timestep if start_point is None else start_point  # 시작 시점을 설정
        batch_size = shape[0]  # 배치 크기 설정
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)  # 초기 샘플 x를 생성 또는 주어진 노이즈로 설정
        cond = cond.to(device)  # 조건(cond)을 장치로 이동

        if return_diffusion:
            diffusion = [x]  # 확산 과정을 저장할 리스트 초기화

        # 시작 시점부터 0까지 반복하여 샘플링 수행
        for i in tqdm(reversed(range(0, start_point))):  # tqdm을 사용하여 진행 상태를 시각화하면서 역순으로 반복
            # 현재 타임스텝을 채움
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)  # 현재 타임스텝 i로 채운 텐서를 생성
            x, _ = self.p_sample(x, cond, timesteps)  # 현재 타임스텝에서 샘플 x를 생성

            if return_diffusion:
                diffusion.append(x)  # 확산 과정을 저장

        if return_diffusion:
            return x, diffusion  # 최종 샘플과 확산 과정을 반환
        else:
            return x  # 최종 샘플만 반환

        
    @torch.no_grad()
    def ddim_sample(self, shape, cond, **kwargs):
        """
        DDIM(Deterministic Diffusion Implicit Models) 방식으로 샘플링을 수행하는 함수입니다.
        주어진 형태(shape)와 조건(cond)에서 샘플링을 수행합니다.
        """
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        # 샘플링을 위한 시간 범위 설정
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] 형태의 시계열 생성
        times = list(reversed(times.int().tolist()))  # 시계열을 역순으로 정렬
        time_pairs = list(zip(times[:-1], times[1:]))  # 시간쌍 생성 [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device=device)  # 초기 샘플 x를 생성 (정규분포 랜덤값)
        cond = cond.to(device)  # 조건(cond)을 장치로 이동

        x_start = None  # 초기 x_start 값을 None으로 설정

        # 시간쌍을 사용하여 반복
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)  # 현재 시간(time)으로 채운 텐서를 생성
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised)  # 모델 예측 수행

            if time_next < 0:
                x = x_start  # time_next가 0보다 작으면 x를 x_start로 설정하고 다음 반복으로 이동
                continue

            alpha = self.alphas_cumprod[time]  # 현재 시간의 alpha 값
            alpha_next = self.alphas_cumprod[time_next]  # 다음 시간의 alpha 값

            # sigma와 c 값 계산
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)  # x와 같은 형태의 노이즈 생성

            # 새로운 x 값 계산
            x = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise
        return x  # 최종 샘플 x를 반환
    
    @torch.no_grad()
    def long_ddim_sample(self, shape, cond, **kwargs):
        """
        DDIM 방식으로 샘플링을 수행하는 함수입니다.
        주어진 형태(shape)와 조건(cond)에서 샘플링을 수행합니다.
        배치 크기가 1인 경우 일반 ddim_sample 함수를 호출합니다.
        """
        # 배치 크기, 장치, 전체 타임스텝 수, 샘플링 타임스텝 수, eta 값 설정
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        if batch == 1:
            return self.ddim_sample(shape, cond)  # 배치 크기가 1이면 일반 ddim_sample 함수를 호출하여 샘플링

        # 샘플링을 위한 시간 범위 설정
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] 형태의 시계열 생성
        times = list(reversed(times.int().tolist()))  # 시계열을 역순으로 정렬
        weights = np.clip(np.linspace(0, self.guidance_weight * 2, sampling_timesteps), None, self.guidance_weight)  # 가이던스 가중치 설정
        time_pairs = list(zip(times[:-1], times[1:], weights))  # 시간쌍 및 가중치 생성 [(T-1, T-2, w1), (T-2, T-3, w2), ..., (1, 0, wn), (0, -1, wn+1)]

        # 초기 샘플 x를 생성 (정규분포 랜덤값)
        x = torch.randn(shape, device=device)
        # 조건(cond)을 장치로 이동
        cond = cond.to(device)

        assert batch > 1  # 배치 크기가 1보다 큰지 확인
        assert x.shape[1] % 2 == 0  # x의 두 번째 차원이 2의 배수인지 확인
        half = x.shape[1] // 2  # x의 두 번째 차원의 절반 크기

        # 초기 x_start 값을 None으로 설정
        x_start = None

        # 시간쌍과 가중치를 사용하여 반복
        for time, time_next, weight in tqdm(time_pairs, desc='sampling loop time step'):
            # 현재 시간(time)으로 채운 텐서를 생성
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            # 모델 예측 수행
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, weight=weight, clip_x_start=self.clip_denoised)

            # time_next가 0보다 작으면 x를 x_start로 설정하고 다음 반복으로 이동
            if time_next < 0:
                x = x_start
                continue

            # 현재 시간의 alpha 값
            alpha = self.alphas_cumprod[time]
            # 다음 시간의 alpha 값
            alpha_next = self.alphas_cumprod[time_next]

            # sigma와 c 값 계산
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            # x와 같은 형태의 노이즈 생성
            noise = torch.randn_like(x)

            # 새로운 x 값 계산
            x = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

            if time > 0:
                # 각 시퀀스의 첫 번째 절반은 이전 시퀀스의 두 번째 절반
                x[1:, :half] = x[:-1, half:]
        return x  # 최종 샘플 x를 반환


    @torch.no_grad()
    def inpaint_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        """
        인페인팅을 수행하는 함수입니다.
        주어진 형태(shape)와 조건(cond)에서 샘플링을 반복하며, 지정된 영역을 조건에 맞게 채웁니다.
        """
        device = self.betas.device  # 모델 파라미터가 있는 장치를 가져옴

        batch_size = shape[0]  # 배치 크기 설정
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)  # 초기 샘플 x를 생성하거나 주어진 노이즈를 사용
        cond = cond.to(device)  # 조건(cond)을 장치로 이동

        if return_diffusion:
            diffusion = [x]  # 확산 과정을 저장할 리스트 초기화

        mask = constraint["mask"].to(device)  # 제약 조건의 마스크를 장치로 이동
        value = constraint["value"].to(device)  # 제약 조건의 값을 장치로 이동

        start_point = self.n_timestep if start_point is None else start_point  # 시작 시점을 설정
        for i in tqdm(reversed(range(0, start_point))):  # tqdm을 사용하여 진행 상태를 시각화하면서 역순으로 반복
            # 현재 타임스텝을 채움
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # i 단계에서 i-1 단계로 샘플링 수행
            x, _ = self.p_sample(x, cond, timesteps)

            # 각 디노이징 단계 사이에 제약 조건 적용
            value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
            x = value_ * mask + (1.0 - mask) * x

            if return_diffusion:
                diffusion.append(x)  # 확산 과정을 저장

        if return_diffusion:
            return x, diffusion  # 최종 샘플과 확산 과정을 반환
        else:
            return x  # 최종 샘플만 반환


    @torch.no_grad()
    def long_inpaint_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        """
        인페인팅을 수행하는 함수입니다.
        주어진 형태(shape)와 조건(cond)에서 샘플링을 반복하며, 지정된 영역을 조건에 맞게 채웁니다.
        배치 크기가 1인 경우 일반 inpaint_loop를 수행합니다.
        """
        device = self.betas.device  # 모델 파라미터가 있는 장치를 가져옴

        batch_size = shape[0]  # 배치 크기 설정
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)  # 초기 샘플 x를 생성하거나 주어진 노이즈를 사용
        cond = cond.to(device)  # 조건(cond)을 장치로 이동

        if return_diffusion:
            diffusion = [x]  # 확산 과정을 저장할 리스트 초기화

        assert x.shape[1] % 2 == 0  # x의 두 번째 차원이 2의 배수인지 확인
        if batch_size == 1:
            # 배치 크기가 1인 경우 일반 p_sample_loop를 수행
            return self.p_sample_loop(
                shape,
                cond,
                noise=noise,
                constraint=constraint,
                return_diffusion=return_diffusion,
                start_point=start_point,
            )
        assert batch_size > 1  # 배치 크기가 1보다 큰지 확인
        half = x.shape[1] // 2  # x의 두 번째 차원의 절반 크기

        start_point = self.n_timestep if start_point is None else start_point  # 시작 시점을 설정
        for i in tqdm(reversed(range(0, start_point))):  # tqdm을 사용하여 진행 상태를 시각화하면서 역순으로 반복
            # 현재 타임스텝을 채움
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # i 단계에서 i-1 단계로 샘플링 수행
            x, _ = self.p_sample(x, cond, timesteps)
            
            # 각 디노이징 단계 사이에 제약 조건 적용
            if i > 0:
                # 각 시퀀스의 첫 번째 절반은 이전 시퀀스의 두 번째 절반
                x[1:, :half] = x[:-1, half:]

            if return_diffusion:
                diffusion.append(x)  # 확산 과정을 저장

        if return_diffusion:
            return x, diffusion  # 최종 샘플과 확산 과정을 반환
        else:
            return x  # 최종 샘플만 반환


    @torch.no_grad()
    def conditional_sample(
        self, shape, cond, constraint=None, *args, horizon=None, **kwargs
    ):
        """
        조건부 샘플링을 수행하는 함수입니다.
        주어진 형태(shape)와 조건(cond)에서 샘플링을 수행합니다.
        
        조건 (conditions) : [ (time, state), ... ]
        """
        device = self.betas.device  # 모델 파라미터가 있는 장치를 가져옴
        horizon = horizon or self.horizon  # 주어진 horizon이 없으면 기본 horizon 사용

        return self.p_sample_loop(shape, cond, *args, **kwargs)  # p_sample_loop 함수를 호출하여 샘플링 수행

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        """
        q(x_t | x_0)를 샘플링하는 함수입니다.
        주어진 초기 상태 x_start와 시간 t에서 샘플을 생성합니다.
        """
        if noise is None:
            noise = torch.randn_like(x_start)  # 노이즈가 주어지지 않으면 x_start와 같은 형태의 노이즈를 생성

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start  # sqrt_alphas_cumprod를 추출하여 x_start에 곱함
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise  # sqrt_one_minus_alphas_cumprod를 추출하여 노이즈에 곱한 후 더함
        )

        return sample  # 생성된 샘플 반환


    def p_losses(self, x_start, cond, t):
        """
        모델의 손실(loss)을 계산하는 함수입니다.
        주어진 초기 상태 x_start, 조건 cond, 시간 t에서 손실을 계산합니다.
        """
        noise = torch.randn_like(x_start)  # x_start와 같은 형태의 노이즈를 생성
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # 노이즈를 추가한 샘플 x_noisy를 생성

        # 복원(reconstruct)
        x_recon = self.model(x_noisy, cond, t, cond_drop_prob=self.cond_drop_prob)  # 모델을 사용하여 x_noisy를 복원
        assert noise.shape == x_recon.shape  # 노이즈와 복원된 x_recon의 형태가 같은지 확인

        model_out = x_recon
        if self.predict_epsilon:
            target = noise  # epsilon을 예측하는 경우 타겟은 노이즈
        else:
            target = x_start  # 그렇지 않은 경우 타겟은 초기 상태 x_start

        # 전체 복원 손실 계산
        loss = self.loss_fn(model_out, target, reduction="none")  # 복원된 출력과 타겟 간의 손실을 계산
        loss = reduce(loss, "b ... -> b (...)", "mean")  # 손실을 배치 차원으로 평균화
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)  # p2 손실 가중치를 적용

        # 접촉(contact)을 나머지 부분과 분리
        model_contact, model_out = torch.split(model_out, (4, model_out.shape[2] - 4), dim=2)
        target_contact, target = torch.split(target, (4, target.shape[2] - 4), dim=2)

        # 속도 손실 계산
        target_v = target[:, 1:] - target[:, :-1]  # 타겟의 속도 계산
        model_out_v = model_out[:, 1:] - model_out[:, :-1]  # 모델 출력의 속도 계산
        v_loss = self.loss_fn(model_out_v, target_v, reduction="none")  # 속도 손실 계산
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")  # 속도 손실을 배치 차원으로 평균화
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)  # p2 손실 가중치를 적용

        # Forward Kinematics(FK) 손실 계산
        b, s, c = model_out.shape
        # unnormalize (비활성화)
        # model_out = self.normalizer.unnormalize(model_out)
        # target = self.normalizer.unnormalize(target)
        # X, Q
        model_x = model_out[:, :, :3]
        model_q = ax_from_6v(model_out[:, :, 3:].reshape(b, s, -1, 6))
        target_x = target[:, :, :3]
        target_q = ax_from_6v(target[:, :, 3:].reshape(b, s, -1, 6))

        # Forward Kinematics(FK) 수행
        model_xp = self.smpl.forward(model_q, model_x)
        target_xp = self.smpl.forward(target_q, target_x)

        fk_loss = self.loss_fn(model_xp, target_xp, reduction="none")  # FK 손실 계산
        fk_loss = reduce(fk_loss, "b ... -> b (...)", "mean")  # FK 손실을 배치 차원으로 평균화
        fk_loss = fk_loss * extract(self.p2_loss_weight, t, fk_loss.shape)  # p2 손실 가중치를 적용

        # 발(foot) 손실 계산
        foot_idx = [7, 8, 10, 11]

        # 모델 자체의 예측과 일치하는 정적 인덱스 찾기
        static_idx = model_contact > 0.95  # N x S x 4 형태의 텐서로, 모델 접촉이 0.95 이상인 인덱스를 찾음
        model_feet = model_xp[:, :, foot_idx]  # 발 위치 (N, S, 4, 3)를 모델의 FK 결과에서 가져옴
        model_foot_v = torch.zeros_like(model_feet)  # 발 속도를 저장할 텐서를 생성하고 0으로 초기화
        model_foot_v[:, :-1] = (
            model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
        )  # 발의 속도를 계산 (N, S-1, 4, 3) 형태로, 시간 축을 따라 위치 변화 계산
        model_foot_v[~static_idx] = 0  # 정적 인덱스가 아닌 경우 속도를 0으로 설정

        # 발 손실 계산
        foot_loss = self.loss_fn(
            model_foot_v, torch.zeros_like(model_foot_v), reduction="none"
        )  # 발 속도와 0 텐서 간의 손실을 계산
        foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")  # 발 손실을 배치 차원으로 평균화

        # 각 손실을 가중치에 따라 합산하여 최종 손실 계산
        losses = (
            0.636 * loss.mean(),
            2.964 * v_loss.mean(),
            0.646 * fk_loss.mean(),
            10.942 * foot_loss.mean(),
        )
        return sum(losses), losses  # 최종 손실과 각 손실을 반환



    def loss(self, x, cond, t_override=None):
        """
        모델의 손실(loss)을 계산하는 함수입니다.
        주어진 입력 x, 조건 cond, 그리고 선택적 시간 t_override에서 손실을 계산합니다.
        """
        batch_size = len(x)  # 입력 x의 배치 크기를 계산
        if t_override is None:
            t = torch.randint(0, self.n_timestep, (batch_size,), device=x.device).long()  # t_override가 주어지지 않으면 무작위 시간 t 생성
        else:
            t = torch.full((batch_size,), t_override, device=x.device).long()  # t_override가 주어지면 해당 값을 사용하여 시간 t 생성
        return self.p_losses(x, cond, t)  # p_losses 함수를 호출하여 손실 계산

    def forward(self, x, cond, t_override=None):
        """
        모델의 순전파(forward) 함수를 정의합니다.
        주어진 입력 x, 조건 cond, 그리고 선택적 시간 t_override에서 손실을 계산하여 반환합니다.
        """
        return self.loss(x, cond, t_override)  # loss 함수를 호출하여 손실 계산 및 반환

    def partial_denoise(self, x, cond, t):
        """
        입력 x를 부분적으로 디노이즈(denoise)하는 함수입니다.
        주어진 입력 x, 조건 cond, 그리고 시간 t에서 디노이즈된 결과를 반환합니다.
        """
        x_noisy = self.noise_to_t(x, t)  # 입력 x에 노이즈를 추가하여 x_noisy 생성
        return self.p_sample_loop(x.shape, cond, noise=x_noisy, start_point=t)  # p_sample_loop 함수를 호출하여 디노이즈된 결과 반환

    def noise_to_t(self, x, timestep):
        """
        주어진 입력 x와 시간 timestep에서 노이즈를 추가한 결과를 반환하는 함수입니다.
        """
        batch_size = len(x)  # 입력 x의 배치 크기를 계산
        t = torch.full((batch_size,), timestep, device=x.device).long()  # 주어진 timestep 값을 사용하여 시간 t 생성
        return self.q_sample(x, t) if timestep > 0 else x  # timestep이 0보다 크면 q_sample 함수를 호출하여 노이즈 추가, 그렇지 않으면 입력 x 반환

    def render_sample(
        self,
        shape,
        cond,
        normalizer,
        epoch,
        render_out,
        fk_out=None,
        name=None,
        sound=True,
        mode="normal",
        noise=None,
        constraint=None,
        sound_folder="ood_sliced",
        start_point=None,
        render=True,
    ):
        """
        샘플을 렌더링하는 함수입니다.
        주어진 형태(shape), 조건(cond), 정규화(normalizer), 에포크(epoch), 렌더링 출력(render_out) 등을 사용합니다.
        """
        if isinstance(shape, tuple):  # shape가 튜플인 경우
            if mode == "inpaint":  # 인페인팅 모드인 경우
                func_class = self.inpaint_loop
            elif mode == "normal":  # 일반 모드인 경우
                func_class = self.ddim_sample
            elif mode == "long":  # 긴 모드인 경우
                func_class = self.long_ddim_sample
            else:
                assert False, "Unrecognized inference mode"  # 인식되지 않은 모드일 경우 오류 발생
            samples = (
                func_class(
                    shape,
                    cond,
                    noise=noise,
                    constraint=constraint,
                    start_point=start_point,
                )
                .detach()  # 텐서에서 그래디언트 추적을 분리
                .cpu()  # 텐서를 CPU로 이동
            )
        else:
            samples = shape  # shape가 텐서인 경우

        samples = normalizer.unnormalize(samples)  # 샘플을 정규화 해제

        if samples.shape[2] == 151:  # 샘플의 마지막 차원이 151인 경우
            sample_contact, samples = torch.split(
                samples, (4, samples.shape[2] - 4), dim=2
            )  # 접촉 정보를 분리
        else:
            sample_contact = None  # 접촉 정보가 없는 경우

        # Forward Kinematics(FK)를 한 번에 수행
        b, s, c = samples.shape
        pos = samples[:, :, :3].to(cond.device)  # 위치 정보를 가져옴
        q = samples[:, :, 3:].reshape(b, s, 24, 6)  # 회전 정보를 가져옴
        q = ax_from_6v(q).to(cond.device)  # 6D 회전 벡터를 axis-angle로 변환

        if mode == "long":  # long 모드인 경우
            b, s, c1, c2 = q.shape
            assert s % 2 == 0  # 시퀀스 길이가 2의 배수인지 확인
            half = s // 2
            if b > 1:
                # 긴 모드의 경우 선형 보간을 사용하여 위치 연결
                fade_out = torch.ones((1, s, 1)).to(pos.device)  # fade_out 텐서 생성
                fade_in = torch.ones((1, s, 1)).to(pos.device)  # fade_in 텐서 생성
                fade_out[:, half:, :] = torch.linspace(1, 0, half)[None, :, None].to(
                    pos.device
                )  # fade_out 텐서의 절반을 선형 보간으로 채움
                fade_in[:, :half, :] = torch.linspace(0, 1, half)[None, :, None].to(
                    pos.device
                )  # fade_in 텐서의 절반을 선형 보간으로 채움

                pos[:-1] *= fade_out  # pos의 절반을 fade_out으로 스케일링
                pos[1:] *= fade_in  # pos의 절반을 fade_in으로 스케일링

                full_pos = torch.zeros((s + half * (b - 1), 3)).to(pos.device)  # 전체 위치 텐서 생성
                idx = 0
                # pos 텐서의 각 슬라이스를 full_pos 텐서에 추가
                for pos_slice in pos:
                    full_pos[idx : idx + s] += pos_slice  # 현재 인덱스부터 시퀀스 길이까지 슬라이스 추가
                    idx += half  # 인덱스를 절반 길이만큼 증가

                # SLERP를 사용하여 관절 각도를 연결
                slerp_weight = torch.linspace(0, 1, half)[None, :, None].to(pos.device)  # SLERP 가중치 생성

                left, right = q[:-1, half:], q[1:, :half]  # 회전 정보를 절반으로 나누어 좌우 생성
                # quaternion으로 변환
                left, right = (
                    axis_angle_to_quaternion(left),  # left 부분을 axis-angle에서 quaternion으로 변환
                    axis_angle_to_quaternion(right),  # right 부분을 axis-angle에서 quaternion으로 변환
                )
                merged = quat_slerp(left, right, slerp_weight)  # SLERP로 보간된 quaternion
                # axis-angle로 다시 변환
                merged = quaternion_to_axis_angle(merged)  # 보간된 quaternion을 다시 axis-angle로 변환

                # 전체 회전 텐서 생성
                full_q = torch.zeros((s + half * (b - 1), c1, c2)).to(pos.device)  # full_q 텐서를 0으로 초기화하여 생성
                full_q[:half] += q[0, :half]  # 첫 번째 절반을 q의 첫 번째 절반으로 설정
                idx = half  # 인덱스를 절반으로 설정
                # 보간된 각도 슬라이스를 full_q에 추가
                for q_slice in merged:
                    full_q[idx : idx + half] += q_slice  # 현재 인덱스부터 절반 길이까지 q_slice를 full_q에 추가
                    idx += half  # 인덱스를 절반 길이만큼 증가
                full_q[idx : idx + half] += q[-1, half:]  # 마지막 절반을 q의 마지막 절반으로 설정

                # FK를 위해 배치 차원 추가
                full_pos = full_pos.unsqueeze(0)  # full_pos에 배치 차원을 추가
                full_q = full_q.unsqueeze(0)  # full_q에 배치 차원을 추가
            else:
                full_pos = pos  # pos가 하나인 경우 그대로 사용
                full_q = q  # q가 하나인 경우 그대로 사용

            # FK를 수행하여 전체 포즈를 계산하고 numpy 배열로 변환
            full_pose = self.smpl.forward(full_q, full_pos).detach().cpu().numpy()  # SMPL 모델을 사용하여 포워드 패스 수행 후 결과를 numpy 배열로 변환
            # 배치 차원을 제거하고 렌더링
            skeleton_render(
                full_pose[0],  # 첫 번째 포즈를 렌더링
                epoch=f"{epoch}",  # 에포크 번호를 문자열로 설정
                out=render_out,  # 렌더링 결과를 저장할 경로
                name=name,  # 렌더링 결과의 이름
                sound=sound,  # 소리 출력 여부
                stitch=True,  # 포즈 연결 여부
                sound_folder=sound_folder,  # 소리 파일 폴더
                render=render  # 렌더링 여부
            )
            if fk_out is not None:  # fk_out 경로가 주어진 경우
                outname = f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.pkl'  # 출력 파일 이름 생성
                Path(fk_out).mkdir(parents=True, exist_ok=True)  # fk_out 경로 생성
                pickle.dump(
                    {
                        "smpl_poses": full_q.squeeze(0).reshape((-1, 72)).cpu().numpy(),  # SMPL 포즈를 numpy 배열로 변환
                        "smpl_trans": full_pos.squeeze(0).cpu().numpy(),  # SMPL 변환을 numpy 배열로 변환
                        "full_pose": full_pose[0],  # 전체 포즈
                    },
                    open(os.path.join(fk_out, outname), "wb"),  # 파일에 덤프
                )
            return

        # q와 pos를 사용하여 FK를 수행하고 결과를 numpy 배열로 변환
        poses = self.smpl.forward(q, pos).detach().cpu().numpy()  # SMPL 모델을 사용하여 포워드 패스 수행 후 결과를 numpy 배열로 변환
        sample_contact = (
            sample_contact.detach().cpu().numpy()
            if sample_contact is not None
            else None
        )  # sample_contact가 None이 아닌 경우 detach() 후 numpy 배열로 변환, None인 경우 그대로 None

        # inner 함수 정의
        def inner(xx):
            num, pose = xx  # num과 pose를 xx에서 추출
            filename = name[num] if name is not None else None  # name이 None이 아닌 경우 해당 인덱스의 파일명, 그렇지 않은 경우 None
            contact = sample_contact[num] if sample_contact is not None else None  # sample_contact가 None이 아닌 경우 해당 인덱스의 접촉 정보, 그렇지 않은 경우 None
            skeleton_render(
                pose,
                epoch=f"e{epoch}_b{num}",  # 에포크 번호와 배치 인덱스를 포함한 문자열
                out=render_out,  # 렌더링 결과를 저장할 경로
                name=filename,  # 렌더링 결과의 이름
                sound=sound,  # 소리 출력 여부
                contact=contact,  # 접촉 정보
            )

        # 병렬 처리를 통해 포즈를 렌더링
        p_map(inner, enumerate(poses))  # 병렬 처리를 통해 각 포즈를 렌더링

        # fk_out이 주어지고 모드가 long이 아닌 경우
        if fk_out is not None and mode != "long":
            Path(fk_out).mkdir(parents=True, exist_ok=True)  # fk_out 경로 생성
            for num, (qq, pos_, filename, pose) in enumerate(zip(q, pos, name, poses)):  # q, pos, name, poses를 함께 반복
                path = os.path.normpath(filename)  # 파일 경로를 표준화
                pathparts = path.split(os.sep)  # 경로를 디렉토리별로 분리
                pathparts[-1] = pathparts[-1].replace("npy", "wav")  # 파일 확장자를 npy에서 wav로 변경
                pathparts[2] = "wav_sliced"  # 경로의 세 번째 부분을 wav_sliced로 변경
                audioname = os.path.join(*pathparts)  # 경로를 다시 결합
                outname = f"{epoch}_{num}_{pathparts[-1][:-4]}.pkl"  # 출력 파일 이름 생성
                pickle.dump(
                    {
                        "smpl_poses": qq.reshape((-1, 72)).cpu().numpy(),  # SMPL 포즈를 numpy 배열로 변환
                        "smpl_trans": pos_.cpu().numpy(),  # SMPL 변환을 numpy 배열로 변환
                        "full_pose": pose,  # 전체 포즈
                    },
                    open(f"{fk_out}/{outname}", "wb"),  # 파일에 덤프
                )
