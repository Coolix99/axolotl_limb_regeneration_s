/*
OpenCL RandomForestClassifier
classifier_class_name = ObjectSegmenter
feature_specification = gaussian_blur=1 difference_of_gaussian=1 laplace_box_of_gaussian_blur=1 sobel_of_gaussian_blur=1
num_ground_truth_dimensions = 3
num_classes = 2
num_features = 4
max_depth = 2
num_trees = 100
feature_importances = 0.3,0.0,0.37,0.33
positive_class_identifier = 2
apoc_version = 0.12.0
*/
__kernel void predict (IMAGE_in0_TYPE in0, IMAGE_in1_TYPE in1, IMAGE_in2_TYPE in2, IMAGE_in3_TYPE in3, IMAGE_out_TYPE out) {
 sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
 const int x = get_global_id(0);
 const int y = get_global_id(1);
 const int z = get_global_id(2);
 float i0 = READ_IMAGE(in0, sampler, POS_in0_INSTANCE(x,y,z,0)).x;
 float i1 = READ_IMAGE(in1, sampler, POS_in1_INSTANCE(x,y,z,0)).x;
 float i2 = READ_IMAGE(in2, sampler, POS_in2_INSTANCE(x,y,z,0)).x;
 float i3 = READ_IMAGE(in3, sampler, POS_in3_INSTANCE(x,y,z,0)).x;
 float s0=0;
 float s1=0;
if(i0<1540.249755859375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<26381.140625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1426.38427734375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1565.05029296875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<25956.41015625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<26265.703125){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<26265.703125){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1540.249755859375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<26031.97265625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1549.87060546875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1540.249755859375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-37622.8359375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i3<25966.6171875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1409.6260986328125){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-41155.9609375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i0<1540.249755859375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<26031.97265625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1540.249755859375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1479.003662109375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-41155.9609375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i2<-41155.9609375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i2<-39501.14453125){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i2<-41269.54296875){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i0<1540.249755859375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<25956.41015625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-38003.171875){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i3<25966.6171875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1542.22412109375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-39569.8828125){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i2<-41341.81640625){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i2<-41155.9609375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i2<-38706.4765625){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i2<-41269.54296875){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i0<1542.22412109375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-41155.9609375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i0<1564.1038818359375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<25956.41015625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1540.249755859375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<26031.97265625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-41155.9609375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i3<26031.97265625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-41155.9609375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i3<26031.97265625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<25956.41015625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1409.6260986328125){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-39010.9609375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i0<1540.249755859375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<25966.6171875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<26031.97265625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<25966.6171875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-39366.39453125){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i2<-41155.9609375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i2<-37836.76171875){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i2<-38116.75390625){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i2<-37436.984375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i3<26031.97265625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-41155.9609375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i0<1424.409912109375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<25966.6171875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<26255.498046875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<25966.6171875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-41269.54296875){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i0<1399.4189453125){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1542.22412109375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1540.249755859375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<26265.703125){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1540.249755859375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-41155.9609375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i2<-41155.9609375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i0<1480.97802734375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1542.22412109375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<24709.296875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-41155.9609375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i3<25956.41015625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-41155.9609375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i0<1479.003662109375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<26031.97265625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-41473.03125){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i0<1448.2640380859375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<25966.6171875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1540.249755859375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<25966.6171875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-39252.81640625){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i2<-39438.66796875){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i3<25966.6171875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1549.87060546875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<25956.41015625){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-41155.9609375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i2<-41155.9609375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i2<-41341.81640625){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i0<1479.003662109375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i3<25966.6171875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-39366.39453125){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i3<25966.6171875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-41155.9609375){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i3<25966.6171875){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i0<1540.249755859375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-39252.81640625){
 s1+=1.0;
} else {
 s0+=1.0;
}
if(i3<24774.65234375){
 s0+=1.0;
} else {
 s1+=1.0;
}
if(i2<-39366.39453125){
 s1+=1.0;
} else {
 s0+=1.0;
}
 float max_s=s0;
 int cls=1;
 if (max_s < s1) {
  max_s = s1;
  cls=2;
 }
 WRITE_IMAGE (out, POS_out_INSTANCE(x,y,z,0), cls);
}
