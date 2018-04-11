from Trainer import Trainer
from TrainerSecond import TrainerSecond
from img_thresholding import img_threshold
from Interlude import Interlude

if __name__ == "__main__":
    process_step = [2, 3, 4]

    full_data_path = '/home/bjh/aneurysm_new_data_train/'
    label_only_data_path = ['/home/bjh/new_train_img_label_only/']

    option_name = '180411-twostep-dice-label_only-batch_off_full_aug'

    # 혈관 라벨 데이터 생성
    if 1 in process_step:
        img_threshold(label_only_data_path)

    if 2 in process_step:
        unet_trainer = Trainer(training_data_path=label_only_data_path, model_path='./model/first_step/Unet.ckpt', validation_percentage=10,
                               initial_learning_rate=0.001, decay_step=2500,
                               decay_rate=0.9, epoch=100, img_size=256,
                               n_class=2, batch_size=50,
                               batch_norm_mode='off', depth=5, option_name=option_name)
        unet_trainer.train()

    if 3 in process_step:
        print('Interlude Step Start!')
        Interluder = Interlude(img_size=256, data_path=label_only_data_path, model_path='./model/first_step/Unet.ckpt',
                               batch_size=20, n_class=2, depth=5)
        Interluder.process()
        print('Interlude Step complete!')

    # 뇌동맥류 학습
    if 4 in process_step:
        second_trainer = TrainerSecond(training_data_path=label_only_data_path, model_path='./model/second_step/Unet.ckpt', validation_percentage=10,
                               initial_learning_rate=0.001, decay_step=2500,
                               decay_rate=0.9, epoch=100, img_size=256,
                               n_class=2, batch_size=50,
                               batch_norm_mode='off', depth=5, option_name=option_name)
        print('Second Step Train Start!')
        second_trainer.train()
        print('Second Step Train complete!')