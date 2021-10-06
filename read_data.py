#!/usr/bin/env python

import tensorflow as tf
import argparse
import sys
import numpy as np
from os import write
from pathlib import Path

from tensorflow.python.summary.summary_iterator import summary_iterator

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def format_latex_string(values, separator='&'):
    s = ''
    for task_k, task_v in values.items():
        s += "Task: {}\n".format(task_k)
        for label_k, label_v in task_v.items():
            s += '{} {} {} ({}) {} {} ({}) {} {} ({}) {}\n'.format(
                label_k, 
                separator,
                label_v['Precision_test_0'],
                label_v['Precision_devel_0'],
                separator,
                label_v['Recall_test_0'],
                label_v['Recall_devel_0'], 
                separator,
                label_v['FScore_test_0'],
                label_v['FScore_devel_0'],
                separator
            )
        s += '\n'
    return s

def format_latex_average_string(averages, separator='&'):
    s = ''
    for task_k, task_v in averages.items():
        s += "Task: {}\n".format(task_k)
        for label_k, label_v in task_v.items():
            s += '{} {} {}$\pm${} {} {}$\pm${} {} {}$\pm${} {} {}$\pm${} {} {}$\pm${} {} {}$\pm${}\\\\ {}\n'.format(
                label_k, 
                separator,
                round(label_v['Precision_test_0']['mean'], 3),
                round(label_v['Precision_test_0']['std'], 3),
                separator,
                round(label_v['Precision_devel_0']['mean'], 3),
                round(label_v['Precision_devel_0']['std'], 3),
                separator,
                round(label_v['Recall_test_0']['mean'], 3),
                round(label_v['Recall_test_0']['std'], 3),
                separator,
                round(label_v['Recall_devel_0']['mean'], 3), 
                round(label_v['Recall_devel_0']['std'], 3), 
                separator,
                round(label_v['FScore_test_0']['mean'], 3),
                round(label_v['FScore_test_0']['std'], 3),
                separator,
                round(label_v['FScore_devel_0']['mean'], 3),
                round(label_v['FScore_devel_0']['std'], 3),
                len(label_v['Precision_test_0']['values'])
            )
        s += '\n'
    return s

def get_epochs(file, args):
    epoch_summary = {}
    for e in summary_iterator(file):
        if e.step >= args.epochs[0] and e.step < args.epochs[1]:
            if e.step not in epoch_summary:
                epoch_summary[e.step] = {}
            for v in e.summary.value:
                if v.tag.count('/') == 3:
                    task, entity, value, dataset = v.tag.split('/')
                else:
                    task = 'default'
                    entity, value, dataset = v.tag.split('/')
                if task not in epoch_summary[e.step]:
                    epoch_summary[e.step][task] = {}
                if entity not in epoch_summary[e.step][task]:
                    epoch_summary[e.step][task][entity] = {}
                epoch_summary[e.step][task][entity][value + '_' + dataset] = v.simple_value
    return epoch_summary

def get_max(file, args):
    max_fscore = 0 
    max_epoch = 0
    #print(file)
    for e in summary_iterator(file):
        for v in e.summary.value:
            if v.tag.count('/') == 3 and v.tag == '{}/{}/FScore/devel_0'.format(args.get_max_task, args.get_max_label):
                if v.simple_value > max_fscore:
                    max_fscore = v.simple_value
                    max_epoch = e.step
            elif v.tag.count('/') == 2 and v.tag == '{}/FScore/devel_0'.format(args.get_max_label):
                if v.simple_value > max_fscore:
                    max_fscore = v.simple_value
                    max_epoch = e.step

    #if e.step < 100:
    #    print("HEappfaefa")
    #    return None
        
    epoch_summary = {
        max_epoch: {}
    }
    for e in summary_iterator(file):
        if e.step == max_epoch:
            for v in e.summary.value:
                if v.tag.count('/') == 3:
                    task, entity, value, dataset = v.tag.split('/')
                else:
                    task = 'default'
                    entity, value, dataset = v.tag.split('/')
                if task not in epoch_summary[e.step]:
                    epoch_summary[e.step][task] = {}
                if entity not in epoch_summary[e.step][task]:
                    epoch_summary[e.step][task][entity] = {}
                epoch_summary[e.step][task][entity][value + '_' + dataset] = v.simple_value
    #print(epoch_summary)
    return epoch_summary

def get_average_performance(files, args):
    averages = {}
    for file in files:
        epoch_summary = get_max(str(file), args) 
        if epoch_summary is None:
            continue
        epoch_result = list(epoch_summary.values())[0]
        for task, task_res in epoch_result.items():
            if task not in averages:
                averages[task] = {}
            for label, label_res in task_res.items():
                if label not in averages[task]:
                    averages[task][label] = {}
                for score, score_res in label_res.items():
                    if score not in averages[task][label]:
                        averages[task][label][score] = {
                            'values': []
                        }
                    averages[task][label][score]['values'].append(score_res)
    for _, task_res in averages.items():
        for _, label_res in task_res.items():
            for score, score_res in label_res.items():
                mean_val = np.mean(score_res['values'])
                score_res['mean'] = mean_val
                std_val = np.std(score_res['values'])
                score_res['std'] = std_val
    return averages

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Get values from tensorboard.")
    parser.add_argument("--path", required=True, help="Full path to tensorboard log or path to folder containing multiple logs")
    parser.add_argument("--get-max", action="store_true", help="Get the max score based on devel F-Score.")
    parser.add_argument("--get-max-task", default="software")
    parser.add_argument("--get-max-label", default="Application")
    parser.add_argument("--epochs", default=[10,12], nargs=2, help="Epochs to search")
    parser.add_argument("--round", default='2', help="How to round results")
    parser.add_argument("--merge-all", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.path)
    if in_path.is_file():
        file_list = {'single': [in_path]}
    elif in_path.is_dir():
        file_list = {}
        events = list(in_path.rglob('events.out.tfevents*'))
        for event in events:
            if args.merge_all:
                config_string = 'all'
            else:
                config_string = str(event).rsplit('/', maxsplit=2)[-2].split('_', maxsplit=2)[-1]
            if config_string not in file_list:
                file_list[config_string] = []
            file_list[config_string].append(event)
    else:
        raise(RuntimeError("Invalid input path {}".format(args.path)))

    for k,v in file_list.items():
        print(k)
        if not args.get_max:
            for file in v:
                args.epochs = [int(x) for x in args.epochs]
                epoch_summary = get_epochs(str(file), args)
                for k,v in epoch_summary.items():
                    print("Epoch {}".format(k))
                    print(format_latex_string(v))
        else:
            averages = get_average_performance(v, args)
            print(format_latex_average_string(averages))
            
        print()
        