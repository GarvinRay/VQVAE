from torch import nn
class MaskedCNN(nn.Conv2d):
  # 继承nn.conv2d 也可以作为成员变量使用.
	def __init__(self, mask_type, *args, **kwargs):
		super(MaskedCNN, self).__init__(*args, **kwargs)
		self.mask_type = mask_type
		self.register_buffer('mask', self.weight.data.clone())

		_, depth, height, width = self.weight.size()
		self.mask.fill_(1)
		if mask_type =='A':
			self.mask[:,:,height//2,width//2:] = 0
			self.mask[:,:,height//2+1:,:] = 0
		else:
			self.mask[:,:,height//2,width//2+1:] = 0
			self.mask[:,:,height//2+1:,:] = 0

	def forward(self, x):
		self.weight.data*=self.mask
		return super(MaskedCNN, self).forward(x)