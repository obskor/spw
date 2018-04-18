from Trainer import Trainer
from TrainerSecond import TrainerSecond
from img_thresholding import img_threshold

if __name__ == "__main__":
    process_step = [2, 3]

    full_data_path = '/home/bjh/aneurysm_new_data_train/'
    label_only_data_path = ['/home/bjh/new_train_img_label_only/']

    option_name = '180418-onestep_pretrain-dice-label_only-bn20-ch12-re_conv'
    use_gpu = "0"

    # 혈관 라벨 데이터 생성
    if 1 in process_step:
        img_threshold(label_only_data_path)

    # 혈관 학습(Pre-Train)
    if 2 in process_step:
        pre_trainer = Trainer(training_data_path=label_only_data_path, model_path='./model/first_step/Unet.ckpt', validation_percentage=10,
                              initial_learning_rate=0.005, decay_step=2500,
                              decay_rate=0.95, epoch=200, img_size=256,
                              n_class=2, batch_size=20, root_channel=12,
                              batch_norm_mode='on', depth=5, option_name=option_name, use_gpu=use_gpu)
        print('Pre-Train Step Start!')
        pre_trainer.train()
        print('Pre-Train Step Complete!')

    # 뇌동맥류 학습
    if 3 in process_step:
        main_trainer = TrainerSecond(training_data_path=label_only_data_path, model_path='./model/first_step/Unet.ckpt', validation_percentage=10,
                                     initial_learning_rate=0.005, decay_step=2500,
                                     decay_rate=0.95, epoch=200, img_size=256,
                                     n_class=2, batch_size=20, root_channel=12,
                                     batch_norm_mode='on', depth=5, option_name=option_name, use_gpu=use_gpu)
        print('Main Step Train Start!')
        main_trainer.train()
        print('Main Step Train complete!')


    # # tensorboard --logdir=./hello_tf_180326-1/