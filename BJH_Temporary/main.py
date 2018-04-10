from Trainer import Trainer
# from Tester import Tester

if __name__ == "__main__":
    full_data_path = '/home/bjh/aneurysm_new_data_train'
    label_only_data_path = '/home/bjh/new_train_img_label_only'

    unet_trainer = Trainer(training_data_path=[label_only_data_path], model_path='./model/Unet.ckpt', validation_percentage=10,
                           initial_learning_rate=0.0001, decay_step=1000,
                           decay_rate=0.95, epoch=200, img_size=256,
                           n_class=2, batch_size=50)
    unet_trainer.train()
    #
    # print('Train complete!')
    # print('Test Start...')
    #
    # unet_tester = Tester(img_size=256, data_path=['D:\\Brain_Aneurysm_dataset\\abnorm\\test_img'],
    #                      model_path='D:\\Git\\anurysm\\local_working\\2d_unet_test\\focal3\\model\\Unet.ckpt', batch_size=10)
    # unet_tester.test()
    #
    # print('Test Complete!')
