def print_result(model):
    print('*****************************************')
    print('Best Parameters found on the training set: {}'.format(model.best_params_))
    print('with accuracy on training set: {0:.2f}%'.format(model.best_score_*100))
    print('*****************************************')
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    y_predict = model.predict(X_val)
    
    for mean, std, params in zip(means, stds, model.cv_results_['params']):
        print('{} {} for {}'.format(mean, std*2, params))
    print('======================================')
    
    global highest_scores
    
    for i in scores:
        if i == 'recall':
            highest_scores.append({'model': model.best_estimator_,
                                  'params': model.best_params_,
                                  'score': model.best_score_,
                                  'Validation_score': accuracy_score(y_val, y_predict)})