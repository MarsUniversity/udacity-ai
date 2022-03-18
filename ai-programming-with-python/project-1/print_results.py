#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/print_results.py
#
# Kevin Walchko
#
# TODO 6: Define print_results function below, specifically replace the None
#       below by the function definition of the print_results function.
#       Notice that this function doesn't to return anything because it
#       prints a summary of the results using results_dic and results_stats_dic
#
def print_results(results_dic, results_stats_dic, model,
                  print_incorrect_dogs = False, print_incorrect_breed = False):
    """
    Prints summary results on the classification and then prints incorrectly
    classified dogs and incorrectly classified dog breeds if user indicates
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
      results_stats_dic - Dictionary that contains the results statistics (either
                   a  percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
      model - Indicates which CNN model architecture will be used by the
              classifier function to classify the pet images,
              values must be either: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and
                             False doesn't print anything(default) (bool)
      print_incorrect_breed - True prints incorrectly classified dog breeds and
                              False doesn't print anything(default) (bool)
    Returns:
           None - simply printing results.
    """
    print("\n\n*** Results Summary for CNN Model Architecture",model.upper(),
          "***")
    print("{:20}: {:3d}".format('N Images', results_stats_dic['n_images']))
    print("{:20}: {:3d}".format('N Dog Images', results_stats_dic['n_dogs_img']))
    print(f"N Not-Dog Images    :  {results_stats_dic['n_notdogs_img']}")
    print(" ")

    # print(results_stats_dic)

    # pct_match - percentage of correct matches
    # pct_correct_dogs - percentage of correctly classified dogs
    # pct_correct_breed - percentage of correctly classified dog breeds
    # pct_correct_notdogs - percentage of correctly classified NON-dogs
    keys = ["pct_match","pct_correct_dogs","pct_correct_breed","pct_correct_notdogs"]
    titles = ["% Match","% Correct Dogs", "% Correct Breed", "% Correct Not Dogs"]
    for key, title in zip(keys,titles):
        print(f"{title}: {results_stats_dic[key]}")

    """
    key = pet image filename (ex: Beagle_01141.jpg)
    value = List with:
    index 0 = Pet Image Label (ex: beagle)
    index 1 = Classifier Label (ex: english foxhound)
    index 2 = 0/1 where 1 = labels match , 0 = labels don't match (ex: 0)
    index 3 = 0/1 where 1= Pet Image Label is a dog, 0 = Pet Image Label isn't a dog (ex: 1)
    index 4 = 0/1 where 1= Classifier Label is a dog, 0 = Classifier Label isn't a dog (ex: 1)
    """
    if (print_incorrect_dogs and
        ( (results_stats_dic['n_correct_dogs'] + results_stats_dic['n_correct_notdogs'])
          != results_stats_dic['n_images'] )
        ):
        print("\nINCORRECT Dog/NOT Dog Assignments:")
        for key in results_dic:
            v = results_dic[key]
            if v[3] != v[4]:
                print(f"REAL: {v[0]}\tCLASSIFIER: {v[1]}")
                # print("\nINCORRECT Dog/NOT Dog Assignments:")
                # print(results_stats_dic['n_correct_dogs'] - results_stats_dic['n_correct_notdogs'])

    if (print_incorrect_breed and
        (results_stats_dic['n_correct_dogs'] != results_stats_dic['n_correct_breed'])
        ):
        print("\nINCORRECT Dog Breed Assignment:")
        # process through results dict, printing incorrectly classified breeds
        for key in results_dic:

            # Pet Image Label is-a-Dog, classified as-a-dog but is WRONG breed
            if ( sum(results_dic[key][3:]) == 2 and
                results_dic[key][2] == 0 ):
                print("Real: {:>26}   Classifier: {:>30}".format(results_dic[key][0],
                                                          results_dic[key][1]))
        # print(results_stats_dic['n_correct_dogs'] - results_stats_dic['n_correct_breed'])
