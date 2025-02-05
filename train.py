from data import *
from model import *
from metrics import *
import csv
import itertools
from tensorflow.keras.callbacks import ModelCheckpoint

if __name__ == "__main__":
    ##  Hyperparameters
    filters = [64, 32]; kernel_sizes = [(2, 2), (3, 3)]
    optimizers = ['adam', 'sgd']; test_loss = []; test_accuracy = []
    pool_size = (2, 2); dense_units = 2; batch_size = 32; epochs = 10
    save_path = '/home/husammm/Desktop/Courses/Python/MLCourse/uni/ECG_QRS/'
    model_path = os.path.join(save_path, "best_cnn_model.h5")

    ##  Data.py
    levels = 6
    save_path_qrs = "QRS/"
    save_path_notqrs = "notQRS/"
    signal_num100 = "100"
    signal_num101 = "101"
    signal_num102 = "102"

    record100 = wfdb.rdrecord(path100, sampto=360)
    record101 = wfdb.rdrecord(path101, sampto=360)
    record102 = wfdb.rdrecord(path102, sampto=360)

    annotation100 = wfdb.rdann(path100,'atr',sampto=360)
    annotation101 = wfdb.rdann(path101,'atr',sampto=360)
    annotation102 = wfdb.rdann(path102,'atr',sampto=360)

    # Qrs_loc_100 = annotation100.sample
    # Qrs_loc_101 = annotation101.sample
    # Qrs_loc_102 = annotation102.sample

    
    data100 = record100.p_signal
    data101 = record101.p_signal
    data102 = record102.p_signal

    ECG1001 = data100[:,0]
    ECG1002 = data100[:,1]
    times = np.arange(len(ECG1001),dtype=float)/record100.fs

    ECG1011 = data101[:,0]
    ECG1012 = data101[:,1]

    ECG1021 = data102[:,0]
    ECG1022 = data102[:,1]

            ##### Call functions #####
    reconstructed_signal100, individual_terms100 = Fourier_series(times, ECG1001, num_terms = 1000)
    samplingFrequency100 = Fourier_transform(record100)
    coeffs1001, coeffs1002, time_values100, timesRealECG100 = Wavelet_transform(ECG1001, ECG1002, levels= levels)
    images100 = waveletImage(path100, levels= levels)
    save_images(save_path_qrs, images100, signal_num100)

    reconstructed_signal101, individual_terms101 = Fourier_series(times, ECG1011, num_terms = 1000)
    samplingFrequency101 = Fourier_transform(record101)
    coeffs1011, coeffs1012, time_values101, timesRealECG101 = Wavelet_transform(ECG1011, ECG1012, levels= levels)
    images101 = waveletImage(path101, levels= levels)
    save_images(save_path_qrs, images101, signal_num101)

    reconstructed_signal102, individual_terms102 = Fourier_series(times, ECG1021, num_terms = 1000)
    samplingFrequency102 = Fourier_transform(record102)
    coeffs1021, coeffs1022, time_values102, timesRealECG102 = Wavelet_transform(ECG1021, ECG1022, levels= levels)
    images102 = waveletImage(path102, levels= levels)
    save_images(save_path_notqrs, images102, signal_num102)

            ##### Visualazition #####
    ### 100
    # visalize_FS(times, ECG1001, individual_terms100, reconstructed_signal100)
    # visalize_FT(times, ECG1001, samplingFrequency100)
    # visualize_wavelet(timesRealECG100, time_values100, ECG1001, ECG1002, coeffs1001, coeffs1002)
    # visualize_IMGwavelet(images100, levels)

    # ### 101
    # visalize_FS(times, ECG1011, individual_terms101, reconstructed_signal101)
    # visalize_FT(times, ECG1011, samplingFrequency101)
    # visualize_wavelet(timesRealECG101, time_values101, ECG1011, ECG1012, coeffs1011, coeffs1012)
    # visualize_IMGwavelet(images101, levels)

    # ### 102
    # visalize_FS(times, ECG1021, individual_terms102, reconstructed_signal102)
    # visalize_FT(times, ECG1021, samplingFrequency102)
    # visualize_wavelet(timesRealECG102, time_values102, ECG1021, ECG1022, coeffs1021, coeffs1022)
    # visualize_IMGwavelet(images102, levels)

    final_images, annot = annotation(images100, images101, images102)
    x_train, x_test, t_train, t_test = split_data(final_images, annot)


        ## Model.py
    heightOfPicture = 150*4 
    widthOfPicture = 5*4
    dimentions = [heightOfPicture, widthOfPicture]

    x_train, x_val, x_test, t_train, t_val, t_test = load_label_data(heightOfPicture, widthOfPicture)
    
    check_point = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    csv_filename = 'hyperparameter_results.csv'
    with open (csv_filename, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Filters', 'Kernel Size', 'Pool size', 'Dense units', 'Batch size', 'Epochs', 'Optimizer', 'Test Loss', 'Test Accuracy'])
    
        for filter, kernel_size, optimizer in itertools.product (filters, kernel_sizes, optimizers):
            
            model = build_model(dimentions, filter, kernel_size, optimizer)
            binary_preds, test_loss, test_accuracy = train_and_evaluate_model(model, x_train, t_train, x_val, t_val, x_test, t_test, check_point)

            csv_writer.writerow([filter, kernel_size, pool_size, dense_units, batch_size, epochs, optimizer, test_loss, test_accuracy])


        ## metrics.py
    conf_matrix = confusion_matrix(t_test, binary_preds)
    metrics = calculate_matrics(conf_matrix)
    i = 0
    for metric, value in metrics.items():
        print(f'{i} - {metric}: {value}')
        i += 1
    AUC_ROC(metrics['Recall'], metrics['Specificity'])