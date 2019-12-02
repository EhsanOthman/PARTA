import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def folds_cross_validation(bottlenck, cross_validation , Folder_path):
    
    
    

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(file, engine='xlsxwriter')
    
    # Convert the dataframe to an XlsxWriter Excel object.
    
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
    loss = []
    loss = df.get('loss')
    val_loss  = []
    val_loss = df.get('val_loss')
    intensities_loss = []
    intensities_loss = df.get('intensities_loss')
    val_intensities_loss  = []
    val_intensities_loss = df.get('val_intensities_loss')
    classes_loss = []
    classes_loss = df.get('classes_loss')
    val_classes_loss  = []
    val_classes_loss = df.get('val_classes_loss')
    intensities_mean_squared_error = []
    intensities_mean_squared_error = df.get('intensities_acc')
    val_intensities_mean_squared_error = []
    val_intensities_mean_squared_error = df.get('val_intensities_acc')
    classes_acc = []
    classes_acc = df.get('classes_acc')
    val_classes_acc = []
    val_classes_acc = df.get('val_classes_acc')
    

    
    df_loss = df["loss"].tolist()
    Loss = np.array([df_loss])
    
    df_val_loss = df["val_loss"].tolist()
    Val_Loss = np.array([df_val_loss])
    
    df_loss_Intensities = df["intensities_loss"].tolist()
    Intensities_loss = np.array([df_loss_Intensities])
    
    df_val_loss_Intensities = df["val_intensities_loss"].tolist()
    Val_Intensities_loss = np.array([df_val_loss_Intensities])
    
    df_classes_loss = df["classes_loss"].tolist()
    Classes_loss = np.array([df_classes_loss])
    
    df_val_classes_loss = df["val_classes_loss"].tolist()
    Val_Classes_loss = np.array([df_val_classes_loss])
    
    df_Regression = df["intensities_intensities_binary_accuracy"].tolist()
    Regression = np.array([df_Regression])
    
    df_val_loss = df["val_intensities_intensities_binary_accuracy"].tolist()
    Val_Regression = np.array([df_val_loss])
    
    df_Classification = df["classes_acc"].tolist()
    Classification = np.array([df_Classification])
    
    df_val_Classification = df["val_classes_acc"].tolist()
    Val_Classification = np.array([df_val_Classification])
    
    
    
    
    cross_validation_results = []

    cross_validation_results.append(np.average(Loss, axis=1))
    cross_validation_results.append(np.average(Val_Loss, axis=1))
    cross_validation_results.append(np.average(Intensities_loss, axis=1))
    cross_validation_results.append(np.average(Val_Intensities_loss, axis=1))
    cross_validation_results.append(np.average(Classes_loss, axis=1))
    cross_validation_results.append(np.average(Val_Classes_loss, axis=1))
    cross_validation_results.append(np.average(Regression, axis=1))
    cross_validation_results.append(np.average(Val_Regression, axis=1))
    cross_validation_results.append(np.average(Classification, axis=1))
    cross_validation_results.append(np.average(Val_Classification, axis=1))
    
    # accuracies
    plt.title('cross_validation')
    plt.plot(cross_validation_results[0][0], label='train acc')
    plt.plot(cross_validation_results[1][0], label='val acc')
    plt.legend()
    plt.show()
      
    
    # loss
    plt.title('cross_validation')
    plt.plot(cross_validation_results[2][0], label='train loss')
    plt.plot(cross_validation_results[3][0], label='val loss')
    plt.legend()
    plt.show() 
    
    res = np.zeros((len(cross_validation_results),len(cross_validation_results[0][0])), dtype=float)
    for i in  range(len(cross_validation_results)):
        for j in range(len (cross_validation_results [i][0])):
           res [i][j] =  cross_validation_results [i][0][j]
           
    return res