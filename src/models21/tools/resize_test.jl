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
