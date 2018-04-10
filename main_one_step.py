###################################################################################################################
# * Created By BJH, LYE
#
# * 뇌 MRI 사진에서 뇌동맥류를 찾기 위한 Two Step 2D U-net Model 구현.
#   - First Step Model
#       - 뇌 MRI 사진에서 혈관 위치를 찾는 모델.
#       - 2D U-Net
#           - X : 뇌 MRI 사진
#           - Y : 혈관 위치가 Binary로 표시된 이미지
#   - Second Step Model
#       - 혈관 사진에서 뇌동맥류를 찾는 모델.
#       - 2D U-Net
#           - X : 혈관 사진
#           - Y : 뇌동맥류의 위치가 Binary로 표시된 이미지
#   - 작업 과정
#       1) img_threshold
#           - Threshold를 이용하여 뇌 MRI 사진에서 혈관 위치만 걸러낸 First Step 학습용 라벨 생성
#       2) First Train
#           - 1)에서 생성한 라벨을 Y, 원본 사진을 X로 하여 First Step Model 학습
#       3) Interlude
#           - First Step Model로 혈관 위치를 뽑아낸 뒤, Second Step Model의 X 및 Y로 사용할 수 있도록 가공한다.
#           - Second Step X : 혈관 위치를 이용해 뇌 MRI 사진에서 혈관 부분만 원본 색상으로 표시된 이미지 생성
#           - Second Step Y : 혈관 위치를 이용해 뇌동맥류 라벨 중 혈관 부분만 남긴 이미지 생성
#       4) Second Train
#           - 3)에서 생성한 X, Y로 Second Step Model 학습
####################################################################################################################
from img_thresholding import img_threshold
from Trainer import Trainer

if __name__ == "__main__":
    data_path = ["./new_data/"]

    first_trainer = Trainer(training_data_path=data_path, step='one_step', validation_percentage=10,
                            initial_learning_rate=0.0001, decay_step=2500,
                            decay_rate=0.9, epoch=200, img_size=256,
                            n_class=2, batch_size=20)
    print('First Step Train Start!')
    first_trainer.train()
    print('First Step Train complete!')
