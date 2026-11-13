import pandas as pd
from sklearn.metrics import confusion_matrix
import shutil

# Copy plots to Desktop as well
shutil.copy2(r"D:\Project\model104_4_classifier\confusion_matrix_model104_4.png", r"C:\Users\anike\Desktop\confusion_matrix_model104_4.png")
shutil.copy2(r"D:\Project\model109_4_cluster\confusion_matrix_model109_4.png", r"C:\Users\anike\Desktop\confusion_matrix_model109_4.png")

# Model 104.4
df104 = pd.read_csv(r"D:\Project\model104_4_classifier\val_predictions_104_4.csv")
classes104 = ["cohesive", "dysplasia", "noncohesive", "dispersive", "normal", "no_feature", "blank"]
cm104 = confusion_matrix(df104["true_label"], df104["pred_label"], labels=classes104)
cm104_df = pd.DataFrame(cm104, index=classes104, columns=classes104)
print("=== MODEL 104.4 CONFUSION MATRIX (7x7) ===")
print(cm104_df.to_string())

# Model 109.4
df109 = pd.read_csv(r"D:\Project\model109_4_cluster\val_predictions_109_4.csv")
classes109 = ["cohesive", "dysplasia", "noncohesive", "dispersive", "normal"]
cm109 = confusion_matrix(df109["true_label"], df109["pred_label"], labels=classes109)
cm109_df = pd.DataFrame(cm109, index=classes109, columns=classes109)
print("\n=== MODEL 109.4 CONFUSION MATRIX (5x5) ===")
print(cm109_df.to_string())

