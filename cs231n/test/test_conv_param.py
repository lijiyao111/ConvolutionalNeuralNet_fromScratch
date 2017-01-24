def Size_Conv(stride_conv, filter_size, H, W, Nbconv):
    P = (filter_size - 1) / 2  # padd
    Hc = (H + 2 * P - filter_size) / stride_conv + 1
    Wc = (W + 2 * P - filter_size) / stride_conv + 1
    width_pool = 2
    height_pool = 2
    stride_pool = 2
    Hp = (Hc - height_pool) / stride_pool + 1
    Wp = (Wc - width_pool) / stride_pool + 1
    if Nbconv == 1:
        return Hp, Wp
    else:
        H = Hp
        W = Wp
        return Size_Conv(stride_conv, filter_size, H, W, Nbconv - 1)

a=Size_Conv(2,7,32,32,3)

print a