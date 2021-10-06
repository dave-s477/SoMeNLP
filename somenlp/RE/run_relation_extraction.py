from somenlp.NER import OutputHandler, DataHandler, Tuner
from . import REmodel, FeatureGenerator

def main(model_config, data_config, time, data_file_ext, ent_file_ext, rel_file_ext):
    """Main function for performing relation extraction training

    Args:
        model_config (dict): configuration for RE model
        data_config (dict): configuration of input data
        time (str): time marker for saving
        data_file_ext (str): identifying extension for automatic file loading
        ent_file_ext (str): identifying extension for automatic file loading
        rel_file_ext (str): identifying extension for automatic file loading
    """
    print("\nSetting up output handler")
    output_handler = OutputHandler(model_config['general']['name'], time, model_config['general']['checkpoint'])
    output_handler.save_json(model_config, name='model_conf')
    output_handler.save_json(data_config, name='data_conf')

    print("\nSetting up data handler")
    data_handler = DataHandler(data_config=data_config, data_file_extension=data_file_ext, label_file_extension=ent_file_ext, relation_file_extension=rel_file_ext, output_handler=output_handler, checkpoint=model_config['general']['checkpoint'],  max_word_length=model_config['general']['max_word_length'], max_sent_length=model_config['general']['max_sentence_length'])
    data_handler.load_data_from_config()
    data_handler.encoding(tags_only=True)
    data_handler.load_input()

    feature_generator = FeatureGenerator(data_handler, model_config['model']['word_embedding'])
    feature_generator.generate_relation_extraction_features()

    print("\nSetting up model")
    model_w = REmodel(model_config['general'], model_config['model'], data_handler, output_handler, model_config['output'])
    if model_config['general']['train']:
        if 'cross_val' in model_config['general'] and model_config['general']['cross_val']:
            model_w.cross_val()
        else:
            model_w.train()
            model_w.test()
            model_w.save()
            #model_w.show_features_importance()
    else:
        model_w.load(model_config['general']['checkpoint'])
        model_w.test()

def tune(config, time, data_file_extension, ent_file_ext, rel_file_ext):
    """Hyper-parameter tuning for RE model

    Args:
        config (dict): configuration including model and data
        time (str): time marker for saving
        data_file_ext (str): identifying extension for automatic file loading
        ent_file_ext (str): identifying extension for automatic file loading
        rel_file_ext (str): identifying extension for automatic file loading
    """
    tuner = Tuner(config, time)
    iterator = tuner.yield_configs()

    for name, data_conf, model_conf in iterator:
        print(model_conf)
        time_name = '{}_{}'.format(time, name)
        print("Training model {}".format(time_name))
        main(model_conf, data_conf, time_name, data_file_extension, ent_file_ext, rel_file_ext)

def predict(model_config, files, prepro, output):
    print("\nSetting up output handler")
    output_handler = OutputHandler(model_config['general']['name'], checkpoint=model_config['general']['checkpoint'])

    print("\nSetting up data handler")
    data_handler = DataHandler(data_files=files, prepro=prepro, output_handler=output_handler, checkpoint=model_config['general']['checkpoint'],  max_word_length=model_config['general']['max_word_length'], max_sent_length=model_config['general']['max_sentence_length'])
    data_handler.encoding(tags_only=True)

    feature_generator = FeatureGenerator(data_handler, model_config['model']['word_embedding'])

    print("\nSetting up model")
    model_w = REmodel(model_config['general'], model_config['model'], data_handler, output_handler, model_config['output'], feature_generator=feature_generator)
    model_w.load(model_config['general']['checkpoint'])
    model_w.predict(output)
