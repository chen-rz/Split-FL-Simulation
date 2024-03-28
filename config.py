# Network configration
SERVER_ADDR= '127.0.0.1'
SERVER_PORT = 24601

K = 5 # Number of devices

# Unique clients order
HOST2IP, CLIENTS_CONFIG, CLIENTS_LIST = {}, {}, []
for _ in range(1, K + 1):
    HOST2IP['Client-' + str(_)] = '127.0.0.1'
    CLIENTS_CONFIG['Client-' + str(_)] = _ - 1
    CLIENTS_LIST.append(str(24999 + _))

# Dataset configration
dataset_name = 'CIFAR10'
home = '..'
dataset_path = home +'/dataset/'+ dataset_name +'/'
N = 50000 # data length


# Model configration
model_cfg = {
	# (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
	'VGG5' : [('C', 3, 32, 3, 32*32*32, 32*32*32*3*3*3), ('M', 32, 32, 2, 32*16*16, 0), 
	('C', 32, 64, 3, 64*16*16, 64*16*16*3*3*32), ('M', 64, 64, 2, 64*8*8, 0), 
	('C', 64, 64, 3, 64*8*8, 64*8*8*3*3*64), 
	('D', 8*8*64, 128, 1, 64, 128*8*8*64), 
	('D', 128, 10, 1, 10, 128*10)]
}
model_name = 'VGG5'
model_size = 1.28
model_flops = 32.902
total_flops = 8488192
split_layer = [2 for _ in range(K)] #Initial split layers
model_len = 7


# FL training configration
R = 5 # FL rounds
LR = 0.01 # Learning rate
B = 100 # Batch size
