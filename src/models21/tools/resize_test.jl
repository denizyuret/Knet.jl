using PyCall, ImageMagick, ImageTransformations, FileIO
imgfile = "ILSVRC2012_val_00000001.JPEG"

tf = pyimport("tensorflow")
torch = pyimport("torch")
Image = pyimport("PIL.Image")
transforms = pyimport("torchvision.transforms")
rdims(a) = permutedims(a, ndims(a):-1:1)
interpolation = Image.BILINEAR
t1 = transforms.Compose([
    transforms.Resize(224; interpolation),     # torch resize image
    transforms.ToTensor(),
])
t2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224; interpolation),     # torch resize tensor
])

# https://github.com/tensorflow/models/blob/master/official/vision/image_classification/resnet/imagenet_preprocessing.py
t3(image, height_width) = tf.compat.v1.image.resize( 
    tf.keras.preprocessing.image.img_to_array(image), # 3/4-D Tensor of shape [batch?, height, width, channels] if default=channels_last
    height_width,
    method=tf.image.ResizeMethod.BILINEAR,
    align_corners=false
)

img = Image.open(imgfile)
img2 = FileIO.load(imgfile)
pilimage = t1(img).numpy()          # (3, 224, 298) = (channels, height, width)
pttensor = t2(img).numpy()
tfresize = t3(img,[224,298]).numpy() ./ 255 |> (x->permutedims(x,(3,1,2)))
jlresize = imresize(img2, (224,298)) |> channelview |> Array{Float32}
@show display(pilimage[1,101:105,101:105])
@show display(pttensor[1,101:105,101:105])
@show display(tfresize[1,101:105,101:105])
@show display(jlresize[1,101:105,101:105])

# resize variants using ImageMagick don't seem to work as well :(
# mobilenet_v2_100_224_pt: 0.7070 (using torchvision.transforms.Resize)
# mobilenet_v2_100_224_pt: 0.6990 (using torchvision.transforms.Resize after ToTensor)
# mobilenet_v2_100_224_pt: 0.7010 (using imresize)
# for i in */*; do convert -resize 256x256^ $i foo.jpg; mv foo.jpg $i; done
# mobilenet_v2_100_224_pt: 0.6930
# for i in */*; do convert -adaptive-resize 256x256^ $i foo.jpg; mv foo.jpg $i; done
# mobilenet_v2_100_224_pt: 0.6920
# for i in */*; do convert -interpolative-resize 256x256^ $i foo.jpg; mv foo.jpg $i; done
# mobilenet_v2_100_224_pt: 0.69
# for i in */*; do convert -interpolate Bilinear -interpolative-resize 256x256^ $i foo.jpg; mv foo.jpg $i; done
# mobilenet_v2_100_224_pt: 0.6940


#= Chasing down source code for transforms.Resize:
/opt/anaconda3/lib/python3.8/site-packages/torchvision/transforms/transforms.py                                                                                       
Resize: interpolation=InterpolationMode.BILINEAR                                                                                                                      
            img (PIL Image or Tensor): Image to be scaled.                                                                                                            
        return F.resize(img, self.size, self.interpolation)                                                                                                           
from . import functional as F                                                                                                                                         
                                                                                                                                                                      
/opt/anaconda3/lib/python3.8/site-packages/torchvision/transforms/functional.py                                                                                       
resize:                                                                                                                                                               
     if not isinstance(img, torch.Tensor):                                                                                                                            
        pil_interpolation = pil_modes_mapping[interpolation]                                                                                                          
        return F_pil.resize(img, size=size, interpolation=pil_interpolation)                                                                                          
     return F_t.resize(img, size=size, interpolation=interpolation.value)                                                                                             
pil_modes_mapping = {                                                                                                                                                 
    InterpolationMode.NEAREST: 0,                                                                                                                                     
    InterpolationMode.BILINEAR: 2,                                                                                                                                    
    InterpolationMode.BICUBIC: 3,                                                                                                                                     
    InterpolationMode.BOX: 4,                                                                                                                                         
    InterpolationMode.HAMMING: 5,                                                                                                                                     
    InterpolationMode.LANCZOS: 1,                                                                                                                                     
}                                                                                                                                                                     
                                                                                                                                                                      
                                                                                                                                                                      
/opt/anaconda3/lib/python3.8/site-packages/torchvision/transforms/functional_tensor.py                                                                                
    from torch.nn.functional import grid_sample, conv2d, interpolate, pad as torch_pad                                                                                
    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [torch.float32, torch.float64])                                                                   
    align_corners = False if interpolation in ["bilinear", "bicubic"] else None                                                                                       
    img = interpolate(img, size=[size_h, size_w], mode=interpolation, align_corners=align_corners)                                                                    
    if interpolation == "bicubic" and out_dtype == torch.uint8:                                                                                                       
        img = img.clamp(min=0, max=255)                                                                                                                               
    img = _cast_squeeze_out(img, need_cast=need_cast, need_squeeze=need_squeeze, out_dtype=out_dtype)                                                                 
                                                                                                                                                                      
/opt/anaconda3/lib/python3.8/site-packages/torchvision/transforms/functional_pil.py                                                                                   
def resize(img, size, interpolation=Image.BILINEAR):                                                                                                                  
return img.resize((ow, oh), interpolation)                                                                                                                            
img has a resize operation. _is_pil_image(img):                                                                                                                       

                                                                                                                                                                     
/opt/anaconda3/lib/python3.8/site-packages/PIL/Image.py                                                                                                               
resize(self, size, resample=BICUBIC, box=None, reducing_gap=None):                                                                                                    
presumably we call this with PIL.Image.BILINEAR                                                                                                                       
        return self._new(self.im.resize(size, resample, box))                                                                                                         
self.im = core.new(mode, size)                                                                                                                                        
    from . import _imaging as core                                                                                                                                    
    # If the _imaging C module is not present, Pillow will not load.                                                                                                  
    # Note that other modules should not refer to _imaging directly;                                                                                                  
    # import Image and use the Image.core variable instead.                                                                                                           
    # Also note that Image.core is not a publicly documented interface,                                                                                               
    # and should be considered private and subject to change.                                                                                                         
https://github.com/python-pillow/Pillow/blob/master/src/_imaging.c                                                                                                    
        imOut = ImagingResample(imIn, xsize, ysize, filter, box);                                                                                                     
filter=bilinear, box=img-size                                                                                                                                         
https://github.com/python-pillow/Pillow/blob/d374015504df2c7f0316154deade1bc1fa5075df/src/libImaging/Resample.c                                                       
return ImagingResampleInner(                                                                                                                                          
        imIn, xsize, ysize, filterp, box, ResampleHorizontal, ResampleVertical);                                                                                      
# filter=&BILINEAR, box=img-size                                                                                                                                      
           case IMAGING_TYPE_UINT8:                                                                                                                                   
                ResampleHorizontal = ImagingResampleHorizontal_8bpc;                                                                                                  
                ResampleVertical = ImagingResampleVertical_8bpc;                                                                                                      
https://github.com/python-pillow/Pillow/blob/d374015504df2c7f0316154deade1bc1fa5075df/src/libImaging/Resample.c#L616                                                  
calls ResampleHorizontal, ResampleVertical                                                                                                                            

=#
