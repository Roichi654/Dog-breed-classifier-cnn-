    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    # Program: Dog Breed Classifier
    # Purpose: Classify pet images using CNNs and calculate stats

    import argparse
    import time
    from os import listdir
    from classifier import classifier

    def get_input_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--dir', type=str, default='pet_images/',
                            help='path to folder of pet images')
        parser.add_argument('--arch', type=str, default='vgg',
                            help='CNN model architecture: resnet, alexnet, vgg')
        parser.add_argument('--dogfile', type=str, default='dognames.txt',
                            help='text file that contains dognames')
        return parser.parse_args()

    def get_pet_labels(image_dir):
        results_dic = dict()
        filenames = listdir(image_dir)
        for filename in filenames:
            if filename[0]!= '.':
                pet_label = ""
                low_filename = filename.lower()
                word_list = low_filename.split('_')
                for word in word_list:
                    if word.isalpha():
                        pet_label += word + "
                pet_label = pet_label.strip()
                if filename not in results_dic:
                    results_dic[filename] = [pet_label]
                else:
                    print("Warning: Duplicate file:", filename)
        return results_dic

    def classify_images(images_dir, results_dic, model):
        for key in results_dic:
            model_label = classifier(images_dir + key, model)
            model_label = model_label.lower().strip()
            if results_dic[key][0] in model_label:
                results_dic[key].extend([model_label, 1])
            else:
                results_dic[key].extend([model_label, 0])

    def adjust_results4_isadog(results_dic, dogfile):
        dognames_dic = dict()
        with open(dogfile, "r") as infile:
            for line in infile:
                line = line.strip()
                if line not in dognames_dic:
                    dognames_dic[line] = 1
        for key in results_dic:
            pet_label = results_dic[key][0]
            classifier_label = results_dic[key][1]
            pet_is_dog = 1 if pet_label in dognames_dic else 0
            classifier_is_dog = 0
            for dog_name in classifier_label.split(','):
                if dog_name.strip() in dognames_dic:
                    classifier_is_dog = 1
                    break
            results_dic[key].extend([pet_is_dog, classifier_is_dog])

    def calculates_results_stats(results_dic):
        stats_dic = {'n_images': 0, 'n_dogs_img': 0, 'n_notdogs_img': 0,
                     'n_match': 0, 'n_correct_dogs': 0,
                     'n_correct_notdogs': 0, 'n_correct_breed': 0}
        for key in results_dic:
            if results_dic[key][2] == 1: stats_dic['n_match'] += 1
            if results_dic[key][3] == 1:
                stats_dic['n_dogs_img'] += 1
                if results_dic[key][4] == 1: stats_dic['n_correct_dogs'] += 1
                if results_dic[key][2] == 1: stats_dic['n_correct_breed'] += 1
            else:
                if results_dic[key][4] == 0: stats_dic['n_correct_notdogs'] += 1
        stats_dic['n_images'] = len(results_dic)
        stats_dic['n_notdogs_img'] = stats_dic['n_images'] - stats_dic['n_dogs_img']
        stats_dic['pct_match'] = (stats_dic['n_match'] / stats_dic['n_images']) * 100.0
        stats_dic['pct_correct_dogs'] = (stats_dic['n_correct_dogs'] / stats_dic['n_dogs_img']) * 100.0
        stats_dic['pct_correct_breed'] = (stats_dic['n_correct_breed'] / stats_dic['n_dogs_img']) * 100.0
        stats_dic['pct_correct_notdogs'] = (stats_dic['n_correct_notdogs'] / stats_dic['n_notdogs_img']) * 100.0 if stats_dic['n_notdogs_img'] > 0 else 0.0
        return stats_dic

    def print_results(results_dic, results_stats, model,
                      print_incorrect_dogs = False, print_incorrect_breed = False):
        print("\n\n*** Results Summary for CNN Model Architecture:", model.upper(), "***")
        print("{:20}: {:3d}".format('N Images', results_stats['n_images']))
        print("{:20}: {:3d}".format('N Dog Images', results_stats['n_dogs_img']))
        print("{:20}: {:3d}".format('N Not-Dog Images', results_stats['n_notdogs_img']))
        print(" ")
        for key in results_stats:
            if key.startswith('pct'):
                print("{:20}: {:.2f}%".format(key, results_stats[key]))

    def main():
        start_time = time.time()
        in_args = get_input_args()
        results = get_pet_labels(in_args.dir)
        classify_images(in_args.dir, results, in_args.arch)
        adjust_results4_isadog(results, in_args.dogfile)
        results_stats = calculates_results_stats(results)
        print_results(results, results_stats, in_args.arch)
        end_time = time.time()
        tot_time = end_time - start_time
        print("\n** Total Elapsed Runtime: {:0.2f} seconds".format(tot_time))

    if __name__ == "__main__":
        main()
