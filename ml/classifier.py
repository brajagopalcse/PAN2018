import pickle, os, sys
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier


class Classification:

    def create_model(self):
        # change the classifiers here
        return SVC(kernel='linear', C=1)
        # return RandomForestClassifier(max_depth=3, random_state=0)

    def ten_fold_cross_validation(self, featuers, labels):
        print('Performing 10 fold cross validation')
        list_of_features = []
        gold_labels = []
        for i_d in featuers:
            list_of_features.append(featuers[i_d])
            gold_labels.append(labels[i_d])
        model = self.create_model()
        fold = 10
        scores = cross_val_score(model, list_of_features, gold_labels, cv=fold)
        print('Accuracy of the {}-fold cross validation is = {}'.format(fold, float(sum(scores))/float(fold)))

    def train(self, featuers, labels):
        list_of_features = []
        gold_labels = []
        for i_d in featuers:
            list_of_features.append(featuers[i_d])
            gold_labels.append(labels[i_d])
        print('Training')
        model = self.create_model()
        model.fit(list_of_features, gold_labels)
        return model

    def test(self, featuers, model):
        labels = {}
        print('Performing test on {} items'.format(len(featuers)))
        list_of_features = []
        keys = []
        for i_d in featuers:
            keys.append(i_d)
            list_of_features.append(featuers[i_d])
        predicted_labels = model.predict(list_of_features)
        for k, v in zip(keys, predicted_labels):
            labels[k] = v
        return labels

    def save_model(self, model, model_name):
        pickle.dump(model, open(model_name, 'wb'))

    def load_model(self, model_name):
        if os.path.exists(os.path.abspath(model_name)):
            return pickle.load(open(model_name, 'rb'))
        else:
            print(model_name + 'is not present for loading')
            sys.exit(1)







