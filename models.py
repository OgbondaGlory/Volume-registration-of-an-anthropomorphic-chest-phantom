# In[1]:
from voxelmorph.torch.networks import VxmDense

# In[2]:

class MyCustomVxmDense(VxmDense):
    def __init__(self, inshape, nb_unet_features):
        super(MyCustomVxmDense, self).__init__(inshape, nb_unet_features)
        # Additional customizations or modifications to the VxmDense model can be done here

# Now you can use MyCustomVxmDense in your code
# model = MyCustomVxmDense(inshape=(256, 256, 256), nb_unet_features=[[32, 64, 128, 256, 512], [512, 256, 128, 64, 32]])
