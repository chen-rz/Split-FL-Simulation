import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import threading
import tqdm
import numpy as np

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.append('../')
from Communicator import *
import utils
import config

np.random.seed(0)
torch.manual_seed(0)

class Sever(Communicator):
	def __init__(self, index, ip_address, server_port, model_name):
		super(Sever, self).__init__(index, ip_address)
		# self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		self.device = 'cpu' # TODO device
		self.port = server_port
		self.model_name = model_name
		self.sock.bind((self.ip, self.port))
		self.client_socks = {}

		while len(self.client_socks) < config.K:
			self.sock.listen(5)
			logger.info("Waiting Incoming Connections.")
			(client_sock, (ip, port)) = self.sock.accept()
			logger.info('Got connection from ' + str(ip) + ':' + str(port))
			logger.info(client_sock)

			# TODO For simulation
			# self.client_socks[str(ip)] = client_sock
			self.client_socks[str(port)] = client_sock

		self.uninet = utils.get_model('Unit', self.model_name, config.model_len-1, self.device, config.model_cfg)

		self.transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
		self.testset = torchvision.datasets.CIFAR10(root=config.dataset_path, train=False, download=False, transform=self.transform_test)
		self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=100, shuffle=False, num_workers=4)
 		
	def initialize(self, split_layers, offload, first, LR):
		if offload or first:
			self.split_layers = split_layers
			self.nets = {}
			self.optimizers= {}
			for i in range(len(split_layers)):
				client_port = config.CLIENTS_LIST[i] # TODO For simulation, port
				if split_layers[i] < len(config.model_cfg[self.model_name]) -1: # Only offloading client need initialize optimizer in server
					self.nets[client_port] = utils.get_model('Server', self.model_name, split_layers[i], self.device, config.model_cfg) # TODO For simulation, port

					#offloading weight in server also need to be initialized from the same global weight
					cweights = utils.get_model('Client', self.model_name, split_layers[i], self.device, config.model_cfg).state_dict()
					pweights = utils.split_weights_server(self.uninet.state_dict(),cweights,self.nets[client_port].state_dict()) # TODO For simulation, port
					self.nets[client_port].load_state_dict(pweights) # TODO For simulation, port

					self.optimizers[client_port] = optim.SGD(self.nets[client_port].parameters(), lr=LR, # TODO For simulation, port
					  momentum=0.9)
				else:
					self.nets[client_port] = utils.get_model('Server', self.model_name, split_layers[i], self.device, config.model_cfg) # TODO For simulation, port
			self.criterion = nn.CrossEntropyLoss()

		msg = ['MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', self.uninet.state_dict()]
		for i in self.client_socks:
			self.send_msg(self.client_socks[i], msg)

	def train(self, thread_number, client_ports): # TODO For simulation, port
		# Network test
		self.net_threads = {}
		for i in range(len(client_ports)): # TODO For simulation, port
			# self.net_threads[client_ports[i]] = threading.Thread(target=self._thread_network_testing, args=(client_ports[i],))
			# self.net_threads[client_ports[i]].start()
			
			self._thread_network_testing(client_ports[i]) # TODO Serial

			logger.info(str(client_ports[i]) + ' network testing start')

		# TODO Serial
		# for i in range(len(client_ports)):
		# 	self.net_threads[client_ports[i]].join()

		self.bandwidth = {}
		for s in self.client_socks:
			msg = self.recv_msg(self.client_socks[s], 'MSG_TEST_NETWORK')
			self.bandwidth[msg[1]] = msg[2]
			logger.info(str(msg[1]) + ' network testing complete')

		# Training start
		self.threads = {}
		for i in range(len(client_ports)): # TODO For simulation, port
			if config.split_layer[i] == (config.model_len -1):
				# self.threads[client_ports[i]] = threading.Thread(target=self._thread_training_no_offloading, args=(client_ports[i],))
				# self.threads[client_ports[i]].start()

				self._thread_training_no_offloading(client_ports[i]) # TODO Serial

				logger.info(str(client_ports[i]) + ' no offloading training start')
			else:
				# self.threads[client_ports[i]] = threading.Thread(target=self._thread_training_offloading, args=(client_ports[i],))
				# self.threads[client_ports[i]].start()

				self._thread_training_offloading(client_ports[i]) # TODO Serial

				logger.info(str(client_ports[i]) + ' offloading training start')

		# TODO Serial
		# for i in range(len(client_ports)):
		# 	self.threads[client_ports[i]].join()

		self.ttpi = {} # Training time per iteration
		for s in self.client_socks:
			msg = self.recv_msg(self.client_socks[s], 'MSG_TRAINING_TIME_PER_ITERATION')
			self.ttpi[msg[1]] = msg[2]

		state = None # No RL

		return state, self.bandwidth

	def _thread_network_testing(self, client_port): # TODO For simulation, port
		msg = self.recv_msg(self.client_socks[client_port], 'MSG_TEST_NETWORK') # TODO For simulation, port
		msg = ['MSG_TEST_NETWORK', self.uninet.cpu().state_dict()]
		self.send_msg(self.client_socks[client_port], msg) # TODO For simulation, port

	def _thread_training_no_offloading(self, client_port): # TODO For simulation, port
		pass

	def _thread_training_offloading(self, client_port): # TODO For simulation, port
		iteration = int((config.N / (config.K * config.B))) # TODO WARNING !!!!!
		for i in range(iteration):
			msg = self.recv_msg(self.client_socks[client_port], 'MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER') # TODO For simulation, port
			smashed_layers = msg[1]
			labels = msg[2]

			# inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
			inputs, targets = smashed_layers, labels # TODO device

			self.optimizers[client_port].zero_grad() # TODO For simulation, port
			outputs = self.nets[client_port](inputs) # TODO For simulation, port
			loss = self.criterion(outputs, targets)
			loss.backward()
			self.optimizers[client_port].step() # TODO For simulation, port

			# Send gradients to client
			msg = ['MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT_'+str(client_port), inputs.grad] # TODO For simulation, port
			self.send_msg(self.client_socks[client_port], msg) # TODO For simulation, port

		logger.info(str(client_port) + ' offloading training end') # TODO For simulation, port
		return 'Finish'

	def aggregate(self, client_ports): # TODO For simulation, port
		self.uninet.to(self.device)
		w_local_list =[]
		for i in range(len(client_ports)): # TODO For simulation, port
			msg = self.recv_msg(self.client_socks[client_ports[i]], 'MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER') # TODO For simulation, port
			if config.split_layer[i] != (config.model_len -1):
				w_local = (utils.concat_weights(self.uninet.state_dict(),msg[1],self.nets[client_ports[i]].state_dict()),config.N / config.K) # TODO For simulation, port
				w_local_list.append(w_local)
			else:
				w_local = (msg[1],config.N / config.K)
				w_local_list.append(w_local)
		zero_model = utils.zero_init(self.uninet).state_dict()
		aggregrated_model = utils.fed_avg(zero_model, w_local_list, config.N)
		
		self.uninet.load_state_dict(aggregrated_model)
		return aggregrated_model

	def test(self, r):
		self.uninet.to(self.device)
		self.uninet.eval()
		test_loss = 0
		correct = 0
		total = 0
		with torch.no_grad():
			# for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(self.testloader)):
			for batch_idx, (inputs, targets) in enumerate(self.testloader): # No tqdm for background-running simulation
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				outputs = self.uninet(inputs)
				loss = self.criterion(outputs, targets)

				test_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()

		acc = 100.*correct/total
		logger.info('Test Accuracy: {}'.format(acc))

		# Save checkpoint.
		torch.save(self.uninet.state_dict(), './'+ config.model_name +'.pth')

		return acc

	def adaptive_offload(self, agent, state):
		# config.split_layer = self.action_to_layer(action) # TODO Configure split layer
		logger.info('Next Round OPs: ' + str(config.split_layer))

		msg = ['SPLIT_LAYERS',config.split_layer]
		self.scatter(msg)
		return config.split_layer

	def reinitialize(self, split_layers, offload, first, LR):
		self.initialize(split_layers, offload, first, LR)

	def scatter(self, msg):
		for i in self.client_socks:
			self.send_msg(self.client_socks[i], msg)
