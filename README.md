## 실행방법

**Student Model Layer 설정**
```
wd_student_config.py

StudentEncoderConfig 클래스의 __init__ 메서드 안에 있는 num_hidden_layers에 원하는 student 모델 레이어 개수 지정
StudentDecoderConfig 클래스의 __init__ 메서드 안에 있는 n_layer에 원하는 student 모델 레이어 개수 지정
```

**Weight Distillation 훈련**
```shell=
# knowledge distillation시 적용하는 일반적인 Loss function
python wd_WdLoss_train.py

# tiny Bert에서 적용한 Loss function
# Out of Memory 문제
python wd_KdLoss_train.py
```
원하는 설정에 맞춰 config.yaml과 각 train.py 스크립트에 있는 data_path, teacher model_checkpoint 수정한 후 사용