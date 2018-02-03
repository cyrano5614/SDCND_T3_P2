# Semantic Segmentation
### Project Introduction

-------------------------------------------------------------------------------

The goal of this project is to label the pixels of a road in images using a Fully Convolutional Network (FCN).

-------------------------------------------------------------------------------

### Project Implementation ###

-------------------------------------------------------------------------------

#### Data ####
The data used in this project to train the model is from Road/Lane Detection Evaluation 2013
Kitti dataset.  The data is divided in to training and testing and with training set, the correct img labe is also provided in color masked format.

#### Training ####
The data was augmented with horizontal flip, vertical flip, gaussian blur, and brightness to make the model more robust to changes in input.
``

        flipper = iaa.Fliplr(1.0) # always horizontally flip each input image
        vflipper = iaa.Flipud(1.0) # vertically flip each input image with 90% probability
        # blurer = iaa.GaussianBlur(3.0) # apply gaussian blur
        blurer = iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
        lighter = iaa.Add((-10, 10), per_channel=0.5) # change brightness of images (by -10 to 10 of original value)
        translater = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}) # translate by -20 to +20 percent (per axis)

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file),
                                            image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file),
                                               image_shape)

                if np.random.random() > 0.7:
                    image = flipper.augment_image(image)
                    gt_image = flipper.augment_image(gt_image)
                if np.random.random() > 0.7:
                    image = vflipper.augment_image(image)
                    gt_image = vflipper.augment_image(gt_image)
                # if np.random() > 0.7:
                #     image = translater(image)
                #     gt_image = translater(gt_image)

                if np.random.random() > 0.7:
                    image = blurer.augment_image(image)
                if np.random.random() > 0.7:
                    image = lighter.augment_image(image)

The flipping was done to both original image and masking image, and the gaussian blur and the brightness was done to only the original image as altering the color of masking image would have had negative impact on the model.

#### Result  ####
[image1]: ./runs/1517670243.5980434(most_recent)/um_000058.png
![alt text][image1]

[image2]: ./runs/1517670243.5980434(most_recent)/um_000081.png
![alt text][image2]

[image3]: ./runs/1517670243.5980434(most_recent)/um_000086.png
![alt text][image3]

[image4]: ./runs/1517670243.5980434(most_recent)/umm_000016.png
![alt text][image4]

[image5]: ./runs/1517670243.5980434(most_recent)/um_000016.png
![alt text][image5]

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
