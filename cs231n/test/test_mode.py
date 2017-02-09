num_layers=10
bn_params = [{'mode': 'train'} for i in range(num_layers - 1)]

mode='test'
for bn_param in bn_params:
    bn_param['mode'] = mode

    print(bn_param)
