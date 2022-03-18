# Image Classification for a City Dog Show

## Project Goal

- Improving your programming skills using Python.

In this project, you will use a created image classifier to identify dog breeds. We ask you to focus on Python and not on the actual classifier.

**Description:**

Your city is hosting a citywide dog show and you have volunteered to help the organizing committee with contestant registration. Every participant that registers must submit an image of their dog along with biographical information about their dog. The registration system tags the images based upon the biographical information.

Some people are planning on registering pets that aren’t actual dogs.

You need to use an already developed Python classifier to make sure the participants are dogs.

Note, you **DO NOT** need to create the classifier. It will be provided to you. You will need to apply the Python tools you just learned to USE the classifier.

**Your Tasks:**

- Using your Python skills, you will determine which image classification algorithm works the "best" on classifying images as "dogs" or "not dogs".
- Determine how well the "best" classification algorithm works on correctly identifying a dog's breed. If you are confused by the term image classifier look at it simply as a tool that has an input and an output. The Input is an image. The output determines what the image depicts. (for example, a dog). Be mindful of the fact that image classifiers do not always categorize the images correctly.
- Time how long each algorithm takes to solve the classification problem. With computational tasks, there is often a trade-off between accuracy and runtime. The more accurate an algorithm, the higher the likelihood that it will take more time to run and use more computational resources to run.

## Key Things

```
results_dic:

key = pet image filename (ex: Beagle_01141.jpg)
value = List with:
index 0 = Pet Image Label (ex: beagle)
index 1 = Classifier Label (ex: english foxhound)
index 2 = 0/1 where 1 = labels match , 0 = labels don't match (ex: 0)
index 3 = 0/1 where 1= Pet Image Label is a dog, 0 = Pet Image Label isn't a dog (ex: 1)
index 4 = 0/1 where 1= Classifier Label is a dog, 0 = Classifier Label isn't a dog (ex: 1)

vairables:

n_images - number of images
n_dogs_img - number of dog images
n_notdogs_img - number of NON-dog images
n_match - number of matches between pet & classifier labels
n_correct_dogs - number of correctly classified dog images
n_correct_notdogs - number of correctly classified NON-dog images
n_correct_breed - number of correctly classified dog breeds
pct_match - percentage of correct matches
pct_correct_dogs - percentage of correctly classified dogs
pct_correct_breed - percentage of correctly classified dog breeds
pct_correct_notdogs - percentage of correctly classified NON-dogs
```

# Results

User `resnet` to debug, it runs *really* fast!

``` bash
./check_images.py --arch resnet

... stuff

*** Results Summary for CNN Model Architecture RESNET ***
N Images            :  40
N Dog Images        :  30
N Not-Dog Images    :  10

% Match: 82.5
% Correct Dogs: 100.0
% Correct Breed: 90.0
% Correct Not Dogs: 90.0

INCORRECT Dog/NOT Dog Assignments:
REAL: cat	CLASSIFIER: norwegian elkhound, elkhound

INCORRECT Dog Breed Assignment:
Real:                     beagle   Classifier:  walker hound, walker foxhound
Real:           golden retriever   Classifier:                       leonberg
Real:             great pyrenees   Classifier:                         kuvasz

** Total Elapsed Runtime: 0:0:3
```
