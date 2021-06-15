import torch

from . import OutputHandler, DataHandler, ModelWrapper, Trainer, Tuner
from somenlp.utils import gpu_setup

def setup_cuda(gpu):
    if gpu and torch.cuda.device_count() > 0:
        GPU = str(gpu_setup.pick_gpu_lowest_memory())
        print("Working on GPU: {}".format(GPU))
        device = torch.device("cuda:{}".format(GPU))
    else:
        print("Working on CPU")
        device = torch.device("cpu")
    return device

def main(model_config, data_config, time, data_file_extension, label_file_extension, feature_file_extension, gpu=True):
    print("Setting up cuda")
    device = setup_cuda(gpu)
    
    print("\nSetting up output handler")
    output_handler = OutputHandler(model_config['general']['name'], time, model_config['general']['checkpoint'])
    output_handler.save_json(model_config, name='model_conf')
    output_handler.save_json(data_config, name='data_conf')

    print("\nSetting up data handler")
    tokenizer = model_config['model']['pretrained']['tokenizer'] if 'pretrained' in model_config['model'] and 'tokenizer' in model_config['model']['pretrained'] else None
    data_handler = DataHandler(data_config, data_file_extension, label_file_extension, feature_file_extension, output_handler=output_handler, checkpoint=model_config['general']['checkpoint'], batch_size=model_config['general']['batch_size'], max_word_length=model_config['general']['max_word_length'], max_sent_length=model_config['general']['max_sentence_length'], tokenizer=tokenizer)
    data_handler.load_data_from_config()
    data_handler.encoding()
    data_handler.load_input()
    data_handler.data_loaders()

    if 'embedding' in model_config['model'] and model_config['model']['embedding']['file']:
        print("\nLoading word embedding")
        emb_weights = data_handler.word_embedding(model_config['model']['embedding'])
    else:
        print('\nNo word embedding is needed for the given model.')
        emb_weights = None
    
    print("\nSetting up model")
    model_w = ModelWrapper(model_config, device, emb_weights, data_handler, output_handler)
    model_w.init_model()

    print("\nSetting up trainer")
    trainer = Trainer(device, model_w, data_handler, output_handler, model_config['training'])
    trainer.train()

    print("\nPerforming a final test of the model")
    trainer._test_model()
        
    print("\nSaving final model")
    model_w.save_checkpoint()

def predict(model_config, files, prepro=True, bio_predicition=True, summary_prediction=True, gpu=True):
    print("Setting up cuda")
    device = setup_cuda(gpu)

    print("\nSetting up output handler")
    output_handler = OutputHandler(model_config['general']['name'], checkpoint=model_config['general']['checkpoint'])

    print("\nSetting up data handler")
    tokenizer = model_config['model']['pretrained']['tokenizer'] if 'pretrained' in model_config['model'] and 'tokenizer' in model_config['model']['pretrained'] else None
    data_handler = DataHandler(data_files=files, prepro=prepro, output_handler=output_handler, checkpoint=model_config['general']['checkpoint'], max_word_length=model_config['general']['max_word_length'], max_sent_length=model_config['general']['max_sentence_length'], tokenizer=tokenizer)
    data_handler.encoding()
    
    if 'embedding' in model_config['model'] and model_config['model']['embedding']['file']:
        print("\nLoading word embedding")
        emb_weights = data_handler.word_embedding(model_config['model']['embedding'])
    else:
        print('\nNo word embedding is needed for the given model.')
        emb_weights = None
    
    print("\nSetting up model")
    model_w = ModelWrapper(model_config, device, emb_weights, data_handler, output_handler)
    model_w.init_model()

    print("\nSetting up trainer")
    trainer = Trainer(device, model_w, data_handler, output_handler, model_config['training'])
    trainer.prediction(bio_predicition, summary_prediction)

def tune(config, time, data_file_extension, label_file_extension, feature_file_extension, gpu=True):
    # switch between systematic and random
    tuner = Tuner(config, time)
    iterator = tuner.yield_configs()

    for name, data_conf, model_conf in iterator:
        time_name = '{}_{}'.format(time, name)
        print("Training model {}".format(time_name))
        main(model_conf, data_conf, time_name, data_file_extension, label_file_extension, feature_file_extension, gpu)
