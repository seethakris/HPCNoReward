import os
from collections import OrderedDict


def ExpAnimalDetails(animalname, classifier_type='Bayes'):
    Detailsdict = OrderedDict()
    # Common Information
    Detailsdict['tracklength'] = 200  # 2m track
    Detailsdict['trackbins'] = 5  # 5cm bins
    Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                'Task2': '2 No Rew',
                                'Task3': '3 Fam Rew',
                                'Task4': '4 Nov Rew'}

    Detailsdict['task_colors'] = {'Task1': '#2c7bb6',
                                  'Task2': '#d7191c',
                                  'Task3': '#b2df8a',
                                  'Task4': '#33a02c',
                                  'Control': '#a6dba0'}
    # Animal Specific Info

    # NR6
    if animalname == 'NR6':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ImagingData/Good_behavior/Dataused/NR6/'

        Detailsdict['task_numframes'] = {'Task1': 20000,
                                         'Task2': 20000,
                                         'Task3': 15000,
                                         'Task4': 15000}

        # # To make sure k-fold validation is equalised
        # Detailsdict['task_framestokeep'] = {'Task1': 11100,
        #                                     'Task2': -2605,
        #                                     'Task3': -2,
        #                                     'Task4': -4}  # Track Parameters

        Detailsdict['task_framestokeep'] = {'Task1': 11100,
                                            'Task2': -2600,
                                            'Task3': -2,
                                            'Task4': -4}  # Track Parameters

        Detailsdict['task_framestokeep_afterendzone'] = {'Task1': -7,
                                                         'Task2': -3,
                                                         'Task3': -8,
                                                         'Task4': -6}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 0,
                                           'Task3': 0,
                                           'Task4': 1}
        Detailsdict['v73_flag'] = 0  # If matfile was saved as v7.3
        Detailsdict['animal'] = animalname

    # NR14
    if animalname == 'NR14':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ImagingData/Good_behavior/Dataused/NR14/'

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000,
                                         'Task3': 20000,
                                         'Task4': 15000}

        # # To make sure k-fold validation is equalised
        # Detailsdict['task_framestokeep'] = {'Task1': -1,
        #                                     'Task2': -118,
        #                                     'Task3': -6,
        #                                     'Task4': -8}  # Track Parameters

        Detailsdict['task_framestokeep'] = {'Task1': -1,
                                            'Task2': -113,
                                            'Task3': -6,
                                            'Task4': -8}  # Track Parameters

        Detailsdict['task_framestokeep_afterendzone'] = {'Task1': -6,
                                                         'Task2': -6,
                                                         'Task3': -5,
                                                         'Task4': -4}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           'Task3': 0,
                                           'Task4': 0}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    # NR21
    if animalname == 'NR21':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ImagingData/Good_behavior/Dataused/NR21/'

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000,
                                         'Task3': 15000,
                                         'Task4': 15000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -8,
                                            'Task2': -111,
                                            'Task3': -165,
                                            'Task4': -113}

        Detailsdict['task_framestokeep_afterendzone'] = {'Task1': -7,
                                                         'Task2': -7,
                                                         'Task3': -2,
                                                         'Task4': -6}

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           'Task3': 0,
                                           'Task4': 0}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    # NR23
    if animalname == 'NR23':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ImagingData/Good_behavior/Dataused/NR23/'

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000,
                                         'Task3': 15000,
                                         'Task4': 15000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -3,
                                            'Task2': -3,
                                            'Task3': -7,
                                            'Task4': -6}

        Detailsdict['task_framestokeep_afterendzone'] = {'Task1': -10,
                                                         'Task2': -7,
                                                         'Task3': -5,
                                                         'Task4': -7}

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           'Task3': 1,
                                           'Task4': 0}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    # NR24
    if animalname == 'NR24':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ImagingData/Good_behavior/Dataused/NR24/'

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000,
                                         'Task3': 25000,
                                         'Task4': 15000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -2,
                                            'Task2': -174,
                                            'Task3': -191,
                                            'Task4': -78}

        Detailsdict['task_framestokeep_afterendzone'] = {'Task1': -10,
                                                         'Task2': -6,
                                                         'Task3': -10,
                                                         'Task4': -10}

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 0,
                                           'Task3': 0,
                                           'Task4': 0}
        Detailsdict['v73_flag'] = 0  # If matfile was saved as v7.3
        Detailsdict['animal'] = animalname

    # CFC4
    if animalname == 'CFC4':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ImagingData/Good_behavior/Dataused/CFC4/'

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000,
                                         'Task3': 15000,
                                         'Task4': 15000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -7,
                                            'Task2': -269,
                                            'Task3': -3,
                                            'Task4': -6}

        Detailsdict['task_framestokeep_afterendzone'] = {'Task1': -10,
                                                         'Task2': -10,
                                                         'Task3': -10,
                                                         'Task4': -1}

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           'Task3': 0,
                                           'Task4': 0}
        Detailsdict['v73_flag'] = 1
        Detailsdict['animal'] = animalname

    # CFC17 - Only did reward and no reward in this guy
    if animalname == 'CFC17':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ImagingData/Good_behavior/Dataused/CFC17/'

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -96,
                                            'Task2': -11}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == 'CFC16':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ImagingData/Good_behavior/Dataused/CFC16/'

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -23,
                                            'Task2': -161}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 2}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == 'CFC19':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ImagingData/Good_behavior/Dataused/CFC19/'

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -10,
                                            'Task2': -2}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == 'NR15':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ImagingData/Good_behavior/Dataused/NR15/'

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -11,
                                            'Task2': -2069}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == 'CFC12':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ImagingData/Good_behavior/Dataused/CFC12/'

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 15000,
                                         }

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -5,
                                            'Task2': -7,
                                            }  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 0,
                                           }
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == 'NR31':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ImagingData/Good_behavior/Dataused/NR31/'

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000,
                                         }

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -8,
                                            'Task2': -2,
                                            }  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           }
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == 'NR32':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ImagingData/Good_behavior/Dataused/NR32/'

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 25000,
                                         }

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -4,
                                            'Task2': -204,
                                            }  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           }
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == 'NR34':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ImagingData/Good_behavior/Dataused/NR34/'

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 25000,
                                         }

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -149,
                                            'Task2': -179,
                                            }  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           }
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    # Create Folder to save results
    if classifier_type == 'Bayes':
        Detailsdict['saveresults'] = os.path.join(Detailsdict['foldername'], 'DecoderResults', 'BayesDecoder')
    elif classifier_type == 'SVM':
        Detailsdict['saveresults'] = os.path.join(Detailsdict['foldername'], 'DecoderResults', 'SVMDecoder')
    else:
        Detailsdict['saveresults'] = os.path.join(Detailsdict['foldername'], 'DecoderResults', 'NNDecoder')
    if not os.path.exists(Detailsdict['saveresults']):
        os.mkdir(Detailsdict['saveresults'])

    return Detailsdict


def LickingAnimal(animalname):
    Detailsdict = OrderedDict()
    # Common Information
    Detailsdict['tracklength'] = 200  # 2m track
    Detailsdict['trackbins'] = 5  # 5cm bins
    Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                'Task2': '2 No Rew'}

    Detailsdict['task_colors'] = {'Task1': '#2c7bb6',
                                  'Task2': '#d7191c'}

    if animalname == 'CFC12':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ImagingData/Good_behavior/Dataused/CFC12/'

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 15000,
                                         }

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -5,
                                            'Task2': -7,
                                            }  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 0,
                                           }
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    # Create Folder to save results
    Detailsdict['saveresults'] = os.path.join(Detailsdict['foldername'], 'DecoderResults', 'BayesDecoder')
    if not os.path.exists(Detailsdict['saveresults']):
        os.mkdir(Detailsdict['saveresults'])

    return Detailsdict


def MultiDaysAnimal(animalname):
    Detailsdict = OrderedDict()
    # Common Information
    Detailsdict['tracklength'] = 200  # 2m track
    Detailsdict['trackbins'] = 5  # 5cm bins
    Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                'Task2': '2 Fam Rew',
                                'Task3': '3 No Rew',
                                'Task4': '4 No Rew',
                                'Task5': '5 Fam Rew'}

    if animalname == 'CFC17':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ImagingData/MultiDay/CFC17/'

        Detailsdict['task_numframes'] = {'Task1': 18000,
                                         'Task2': 15000,
                                         'Task3': 20000,
                                         'Task4': 20000,
                                         'Task5': 15000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -10,
                                            'Task2': -4,
                                            'Task3': -4,
                                            'Task4': -9,
                                            'Task5': -3}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 0,
                                           'Task3': 1,
                                           'Task4': 1,
                                           'Task5': 0}

        Detailsdict['NoRewardTasks'] = ['Task3', 'Task4']
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname
        # Create Folder to save results
        Detailsdict['saveresults'] = os.path.join(Detailsdict['foldername'], 'DecoderResults', 'BayesDecoder')
        if not os.path.exists(Detailsdict['saveresults']):
            os.mkdir(Detailsdict['saveresults'])

        return Detailsdict


def ControlAnimals(animalname):
    Detailsdict = OrderedDict()
    # Common Information
    Detailsdict['tracklength'] = 200  # 2m track
    Detailsdict['trackbins'] = 5  # 5cm bins
    Detailsdict['task_dict'] = {'Task1a': '1 Fam Rew',
                                'Task1b': '2 Fam Rew'}

    Detailsdict['task_colors'] = {'Task1a': '#2c7bb6',
                                  'Task1b': '#d7191c'}
    # Animal Specific Info
    if animalname == 'CFC3':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ControlData/Dataused/CFC3/'
        Detailsdict['task_numframes'] = {'Task1a': 15000,
                                         'Task1b': 15000}
        Detailsdict['task_framestokeep'] = {'Task1a': -5,
                                            'Task1b': -6}

        Detailsdict['trackstart_index'] = {'Task1a': 0,
                                           'Task1b': 0}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == 'CFC4':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ControlData/Dataused/CFC4/'
        Detailsdict['task_numframes'] = {'Task1a': 20000,
                                         'Task1b': 20000}
        Detailsdict['task_framestokeep'] = {'Task1a': -3,
                                            'Task1b': -3}

        Detailsdict['trackstart_index'] = {'Task1a': 1,
                                           'Task1b': 1}
        Detailsdict['v73_flag'] = 1
        Detailsdict['animal'] = animalname

    if animalname == 'CFC17':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ControlData/Dataused/CFC17/'
        Detailsdict['task_numframes'] = {'Task1a': 15000,
                                         'Task1b': 15000}
        Detailsdict['task_framestokeep'] = {'Task1a': -5,
                                            'Task1b': -11}

        Detailsdict['trackstart_index'] = {'Task1a': 0,
                                           'Task1b': 1}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == 'CFC19':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ControlData/Dataused/CFC19/'
        Detailsdict['task_numframes'] = {'Task1a': 15000,
                                         'Task1b': 15000}
        Detailsdict['task_framestokeep'] = {'Task1a': -1,
                                            'Task1b': -3}

        Detailsdict['trackstart_index'] = {'Task1a': 0,
                                           'Task1b': 0}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == 'NR31':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ControlData/Dataused/NR31/'
        Detailsdict['task_numframes'] = {'Task1a': 10000,
                                         'Task1b': 10000}
        Detailsdict['task_framestokeep'] = {'Task1a': -6,
                                            'Task1b': -3}

        Detailsdict['trackstart_index'] = {'Task1a': 0,
                                           'Task1b': 0}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == 'NR32':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ControlData/Dataused/NR32/'
        Detailsdict['task_numframes'] = {'Task1a': 10000,
                                         'Task1b': 10000}
        Detailsdict['task_framestokeep'] = {'Task1a': -5,
                                            'Task1b': -5}

        Detailsdict['trackstart_index'] = {'Task1a': 0,
                                           'Task1b': 0}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == 'NR34':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/ControlData/Dataused/NR34/'
        Detailsdict['task_numframes'] = {'Task1a': 10000,
                                         'Task1b': 10000}
        Detailsdict['task_framestokeep'] = {'Task1a': -5,
                                            'Task1b': -4}

        Detailsdict['trackstart_index'] = {'Task1a': 0,
                                           'Task1b': 0}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    # Create Folder to save results
    Detailsdict['saveresults'] = os.path.join(Detailsdict['foldername'], 'DecoderResults', 'BayesDecoder')
    if not os.path.exists(Detailsdict['saveresults']):
        os.mkdir(Detailsdict['saveresults'])

    return Detailsdict
