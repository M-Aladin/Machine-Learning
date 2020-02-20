for i in models.keys():
    for j in scores:
        model = GridSearchCV(estimator = models[i]['estimator'], param_grid=models[i]['param'],
                            scoring = '%s_macro' % j, iid=False)
        model.fit(X_train_t, y_train_t)
        print('Training the {} model'.format(models[i]['name'],'\n'))
        print('Tuning Hyper-parameters for {}'.format(j))
        print_result(model)