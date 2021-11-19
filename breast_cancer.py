import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer

breast_cancer_data = load_breast_cancer()


#print(breast_cancer_data.data[0])
#print(breast_cancer_data.feature_names)

print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)
