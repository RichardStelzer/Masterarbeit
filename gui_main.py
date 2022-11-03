# -----------------------------------------------------------
#
# GUI to visualize results and control subtasks of pipeline
#
# -----------------------------------------------------------

import PySimpleGUI as sg
import os
import functionLibrary as fl
from configobj import ConfigObj
import pyperclip


def write_config(fname, config_integration, config_job=None):
        
    config = ConfigObj()
    config.filename = fname
    
    # Base directories
    config["directories"] = {
        "Img_Folder": "Data/jobs/",
        "Res_Folder": "Data/post/acquisitions/matched/results/",  # "results", #"results_dummies",  # "../to_check"
        "Time_Folder": "Data/post/acquisitions/matched/time/",  # time",
        "Res_Folder_questions": "Data/post/questions/results/",  # "results", #"results_dummies",  # "../to_check"
        "Time_Folder_questions": "Data/post/questions/time_clicks/",  # "results", #"results_dummies",  # "../to_check"
        "GT_Folder": "Data/GT/",  # "GT"
        "Figures": "figures/"
    }
    
    config["campaign-general"] = {
        "categoryId": "09",
        "groupId": "18f477774339",
        "maxPositionPerWorker": 1,
        "qtRequired": True,
        "paymentPerTask": 0.10,
        "height": "500",
        "width": "100%",
        # "maximumJobLimit_enabled": True,
        # "maximumJobLimit_limitPerDay": 1,
        "ttr": 7,
        "description": "Please read the instructions carefully before answering questions."
    }
    
    config["campaign-acquisitions"] = {
        "url": "https://geoinf-rs.bplaced.net/Crowdinterface_Acquisitions?worker={{MW_ID}}&campaign={{CAMP_ID}}&slot={{SLOT_ID}}&rand_key={{RAND_KEY}}",
        "minutesToFinish": 10,
        "title": "Mark Cars in Aerial Image Strip"
    }
    
    config["campaign-questions"] = {
        "url": "https://geoinf-rs.bplaced.net/Crowdinterface_Questions?worker={{MW_ID}}&campaign={{CAMP_ID}}&slot={{SLOT_ID}}&rand_key={{RAND_KEY}}",
        "minutesToFinish": 10,
        "title": "Marked Car in an Aerial Photo Strip: Answer Questions"
    }
    
    # Integration
    if config_integration:
        config["integration"] = {
            "DBSCAN-1": {
                "minpts": config_integration["DBSCAN-1"]["minpts"],
                "epsilon": config_integration["DBSCAN-1"]["epsilon"]   # [pix]
            },
            "DBSCAN-2": {
                "minpts": config_integration["DBSCAN-2"]["minpts"],
                "epsilon": config_integration["DBSCAN-2"]["epsilon"]    # [pix]
            },
            "cellSize": 0.1,  # [m]
            "minimal_length": config_integration["minimal_length"],  # minimal length of acquisition
            "max_len_deviation": config_integration["max_len_deviation"],
            "max_dist_2_integrated_line": config_integration["max_dist_2_integrated_line"],
            "max_distance_correspondence": config_integration["max_distance_correspondence"],
            "minpts_threshold_ellipse_1": 5,
            "std_threshold": 3.0,
            "minpts_threshold_ellipse_2": 8,
            "overwrite_crowd_input": False
        }
    else:  # Standard settings
        config["integration"] = {
            "DBSCAN-1": {
                "minpts": 4,
                "epsilon": 5    # [pix]
            },
            "DBSCAN-2": {
                "minpts": 2,
                "epsilon": 10    # [pix]
            },
            "cellSize": 0.1,  # [m]
            "minimal_length": 8, # [pix] minimal length of acquisition
            "max_len_deviation": 10,
            "max_dist_2_integrated_line": 20,
            "max_distance_correspondence": 10,            
            "minpts_threshold_ellipse_1": 5,
            "std_threshold": 3.0,
            "minpts_threshold_ellipse_2": 8,
            "overwrite_crowd_input": False
        }
    
    # Job settings
    number_of_jobs = sum(os.path.isdir(os.path.join(config["directories"]["Img_Folder"], i)) for i in os.listdir(config["directories"]["Img_Folder"]))
    
    if config_job:
        number_of_acquisitions = config_job["number_of_acquis"]
    else:
        number_of_acquisitions = 25  # standard
        
    config["jobs"] = {
        "number_of_acquisitions": number_of_acquisitions,  # excluding acquisition of augmentation
        "number_of_jobs": number_of_jobs,
        "url_admin": "https://geoinf-rs.bplaced.net/Admininterface"
    }
    
    # Rating settings
    config["rating"] = {
        "step": 5
    }
    
    # Crowdinterface questions
    config["interface_questions"] = {
        "it_numb": 5
    }
    
    # Microworkers
    config["microworkers"] = {
        "api_key": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",  # personal microworkers api key
        "api_url": "https://ttv.microworkers.com/api/v2"   # base api url
    }

    # FTP
    config["ftp"] = {
        "url": "geoinf-rs.bplaced.net",
        "user": "geoinf-rs",
        "passwd": "placeholder"
    }
    
    # Backup
    config["backup"] = {
        "cars_in_shds_pre_verification": "backup/variables/pre-interface-verification____cars_in_shds.dat",
        "cars_in_shds": "backup/variables/cars_in_shds.dat",
        "ellRatingResult": "backup/variables/ellRatingResult.dat",
        "dbRatingResult": "backup/variables/dbRatingResult.dat",
        "ellRatingResult_crowd": "backup/variables/ellRatingResult_crowd.dat",
        "dbRatingResult_crowd": "backup/variables/dbRatingResult_crowd.dat",
        "worker_rating_admin": "backup/variables/worker_rating_admin.dat",
        "worker_rating_crowd": "backup/variables/worker_rating_crowd.dat",
        "worker_rating_ready4submit_admin": "backup/variables/ready4Submit/worker_rating_admin.dat",
        "worker_rating_ready4submit_crowd": "backup/variables/ready4Submit/worker_rating_crowd.dat",
    }

    config.write()


def read_config(fname):
    """
    """
    print("Loading config from = {}".format(fname))
    config = ConfigObj(fname)
    config_return = {}
    
    # Base directories
    config_return["directories"] = config["directories"]

    # Campaign settings
    config_return["campaign-general"] = {
        'categoryId': config["campaign-general"]["categoryId"],
        'groupId': config["campaign-general"]["groupId"],
        'maxPositionPerWorker': config["campaign-general"].as_int("maxPositionPerWorker"),
        'qtRequired':   config["campaign-general"].as_bool("qtRequired"),
        'paymentPerTask':   config["campaign-general"].as_float("paymentPerTask"),
        'height':   config["campaign-general"]["height"],
        'width':    config["campaign-general"]["width"],
        # 'maximumJobLimit_enabled':  config["campaign-general"].as_bool("maximumJobLimit_enabled"),
        # maximumJobLimit_enabled = True
        # maximumJobLimit_limitPerDay = 1
        # 'maximumJobLimit_limitPerDay':  config["campaign-general"].as_int("maximumJobLimit_limitPerDay"),
        'ttr':  config["campaign-general"].as_int("ttr"),
        'description':  config["campaign-general"]["description"],
    }

    config_return["campaign-acquisitions"] = {
        "url":  config["campaign-acquisitions"]["url"],
        "minutesToFinish":  config["campaign-acquisitions"].as_int("minutesToFinish"),
        "title":    config["campaign-acquisitions"]["title"]
    }
    config_return["campaign-questions"] = {
        "url":  config["campaign-questions"]["url"],
        "minutesToFinish":  config["campaign-questions"].as_int("minutesToFinish"),
        "title":    config["campaign-questions"]["title"]
    }

    # Integration params
    integration_params = config["integration"]
    
    config_return["integration"] = {
        "DBSCAN-1": {
            "minpts": integration_params["DBSCAN-1"].as_int("minpts"),
            "epsilon": integration_params["DBSCAN-1"].as_int("epsilon")
        },
        "DBSCAN-2": {
            "minpts": integration_params["DBSCAN-2"].as_int("minpts"),
            "epsilon": integration_params["DBSCAN-2"].as_int("epsilon")
        },
        "cellSize": integration_params.as_float("cellSize"),
        "minimal_length": integration_params.as_float("minimal_length"),
        "max_len_deviation": integration_params.as_float("max_len_deviation"),
        "max_dist_2_integrated_line": integration_params.as_float("max_dist_2_integrated_line"),
        "max_distance_correspondence": integration_params.as_float("max_distance_correspondence"),        
        "minpts_threshold_ellipse_1": integration_params.as_int("minpts_threshold_ellipse_1"),
        "std_threshold": integration_params.as_float("std_threshold"),
        "minpts_threshold_ellipse_2": integration_params.as_int("minpts_threshold_ellipse_2"),
        "overwrite_crowd_input": integration_params.as_bool("overwrite_crowd_input"),
    }
    
    # Job settings    
    config_return["jobs"] = { 
        "number_of_acquisitions": config["jobs"].as_int( "number_of_acquisitions" ),
        "number_of_jobs": config["jobs"].as_int( "number_of_jobs" ),
        "url_admin": config["jobs"]["url_admin"]
    }
    
    config_return["rating"] = {
        "step": config["rating"].as_int( "step" )
    }
    
    config_return["interface_questions"] = {
        "it_numb": config["interface_questions"].as_int( "it_numb" )
    }
    
    config_return["microworkers"] = {
        "api_key":  config["microworkers"]["api_key"],
        "api_url":  config["microworkers"]["api_url"]
    }
    
    config_return["ftp"] = {
        "url":  config["ftp"]["url"],
        "user":  config["ftp"]["user"],
        "passwd":  config["ftp"]["passwd"]        
    }
    
    config_return["backup"] = {
        "cars_in_shds_pre_verification": config["backup"]["cars_in_shds_pre_verification"],
        "cars_in_shds": config["backup"]["cars_in_shds"],
        "ellRatingResult": config["backup"]["ellRatingResult"],
        "dbRatingResult": config["backup"]["dbRatingResult"],
        "ellRatingResult_crowd": config["backup"]["ellRatingResult_crowd"],
        "dbRatingResult_crowd": config["backup"]["dbRatingResult_crowd"],
        "worker_rating_admin": config["backup"]["worker_rating_admin"],
        "worker_rating_crowd": config["backup"]["worker_rating_crowd"],
        "worker_rating_ready4submit_admin": config["backup"]["worker_rating_ready4submit_admin"],
        "worker_rating_ready4submit_crowd": config["backup"]["worker_rating_ready4submit_crowd"]
    }
    
    return config_return


def main_gui():
    """ 
    GUI to visualize results and control subtasks of pipeline
    """
    
    config = read_config("Config/config.ini")
    camp_setting_list = []
    for setting in config["campaign-general"]:        
        camp_setting_list.append( setting + ": " + str(config["campaign-general"][setting]))
    for setting in config["campaign-acquisitions"]:
        line2append = setting + ": " + str(config["campaign-acquisitions"][setting])
        if len(line2append) > 50:
            camp_setting_list.append(line2append[:len(line2append)//2])
            camp_setting_list.append(line2append[len(line2append)//2:])
        else:
            camp_setting_list.append(line2append)
    
    layout_left = [
        # General Settings
        [sg.Text("1. Campaign settings for car acquisition:", font="bold")],
        [sg.Text("Number of acquisitions/workers per image strip = "), sg.In( default_text=config["jobs"]["number_of_acquisitions"], size=(5, 1), enable_events=True, key="number_of_acquis")],
        [sg.Button('Save config', key="saveConfig", bind_return_key=True), sg.Button('Create campaign / submit campaign to microworkers', key="create_campaign_acqui", bind_return_key=True)],
        # [sg.MLine(default_text=camp_setting_list, key='campaign_settings', size=(65,4))]
        [sg.Listbox( values=camp_setting_list, enable_events=False, size=(70, 4), key="campaign_settings")],
        [sg.Text('−'*64)],      # Horizontal Separator
        
        [sg.Text("2. Active car acquisition campaigns:", font="bold")],       
        [sg.Listbox( values=[], enable_events=False, size=(70, 2), key="campaign_list")],
        [sg.Button('Search', key="refresh_active_campaigns", bind_return_key=True, )], 
        [sg.Text('−'*64)],      # Horizontal Separator
        
        # Download (via FTP) & validate gathered data
        [sg.Text("3. Download results of the finished car acquisition campaign from the\n    bplaced server & verify the data by matching it with\n    data stored on the side of microworkers:", font="bold")],   
        [sg.Button('Download & match', key="ftp_download_match", bind_return_key=True)],
        [sg.Text('−'*64)],      # Horizontal Separator
        
        # Integration Settings
        [sg.Text("4. Integration settings:", font="bold", justification='left')],
        [sg.Text("  - DBSCAN 1st iteration:")],    
        [sg.Text("      minPts  ="),
         sg.In(default_text=config["integration"]["DBSCAN-1"]["minpts"], size=(5, 1), enable_events=True, key="dbscan_minPts"),
         sg.Text("epsilon  ="),
         sg.In(default_text=config["integration"]["DBSCAN-1"]["epsilon"], size=(5, 1), enable_events=True, key="dbscan_epsilon"),
         sg.Text("[px]")],
        [sg.Text("  - DBSCAN 2nd iteration:")],
        [sg.Text("      minPts  ="),
         sg.In(default_text=config["integration"]["DBSCAN-2"]["minpts"], size=(5, 1), enable_events=True, key="dbscan_weak_minPts"),
         sg.Text("epsilon  ="),
         sg.In(default_text=config["integration"]["DBSCAN-2"]["epsilon"], size=(5, 1), enable_events=True, key="dbscan_weak_epsilon"),
         sg.Text("[px]")],
        [sg.Text("  - Minimal axis length  ="),
         sg.In(default_text=config["integration"]["minimal_length"], size=(5, 1), enable_events=True, key="min_axis_length"),
         sg.Text("[px]")],
        [sg.Text("  - Max len deviation  ="),
         sg.In(default_text=config["integration"]["max_len_deviation"], size=(5, 1), enable_events=True, key="max_len_deviation"),
         sg.Text("[px]")],
        [sg.Text("  - Max distance to integrated line  ="),
         sg.In(default_text=config["integration"]["max_dist_2_integrated_line"], size=(5, 1), enable_events=True, key="max_dist_2_integrated_line"),
         sg.Text("[px]")],
        [sg.Text("  - Max distance for correspondence  ="),
         sg.In(default_text=config["integration"]["max_distance_correspondence"], size=(5, 1), enable_events=True, key="max_distance_correspondence"),
         sg.Text("[px]")],
        [sg.Text(" ")],
        [sg.Text("Load/Save settings:", font="bold")],
        [sg.Text("Select config file:"),
         sg.In(size=(45, 1), default_text=os.getcwd() + "\config\config.ini", enable_events=True, key="path_config"),
         sg.FileBrowse( initial_folder= os.getcwd() + "\config" , file_types=(("Text Files", "*.ini"),))],
        [sg.Button('Load', key="loadConfig", bind_return_key=True),
         sg.Button('Save', key="saveConfigIntegration", bind_return_key=True)
         ],
        [sg.Text('−'*64)],      # Horizontal Separator
        [sg.Text("5. Start integration (integrate results of the 1st campaign (car acquisition) \n    & generate textfiles containing the \"uncertain\" acquisitions):", font="bold")], 
        [sg.Button('Start integration', key="start_integration", bind_return_key=True)],
    ]

    layout_right = [
        # Download FTP
        [sg.Text("6. Upload the generated textfiles to the server \n    & Remove existing textfiles (previous campaign data):", font="bold")],   
        [sg.Button('Upload to bplaced.net', key="ftp_upload_txt", bind_return_key=True)],   
             
        [sg.Text('−'*64)],
        
        [sg.Text("7. Adminrate", font="bold"), sg.Button('7.1 Generate URLs', key="generate_links", bind_return_key=True)],
        [sg.Listbox(values=[], enable_events=True, size=(70, 3), key="listbox_links")],
        [sg.Button('7.2 Open all links in new chrome window', key="open_browser", bind_return_key=True)],
        [sg.Text('Or click on list item to copy url to the clipboard')],
        [sg.Text('(Strg + F5 to clear cache. Set Reference -> Submit -> Close.)')],
        
        [sg.Text('−'*64)], 
        
        [sg.Text("8. Download admin rating results:", font="bold")],   
        [sg.Button('Download from bplaced.net', key="ftp_download_admin", bind_return_key=True)],   
        
        [sg.Text('−'*64)], 
        
        [sg.Text("9. Create verification campaign (questionnaire):", font="bold")],   
        [sg.Button('Create campaign / submit campaign to microworkers', key="create_campaign_questions", bind_return_key=True)],
        [sg.Text('−'*64)], 
        
        [sg.Text("10. Active verification campaigns (questionnaire):", font="bold")],       
        [sg.Listbox(values=[], enable_events=False, size=(70, 2), key="campaign_list_questions")],
        [sg.Button('Search', key="refresh_active_campaigns_questions", bind_return_key=True, )], 
        [sg.Text('−'*64)], 
        
        [sg.Text("11. Download results of the finished verification campaign (questionnaire):", font="bold")],   
        [sg.Button('Download from bplaced.net', key="ftp_download_crowd", bind_return_key=True)],
        
        [sg.Text('−'*64)],
        
        [sg.Text("12. Calculate ratings for crowdworker of the 1st campaign (1.) using the\n      results from the admininterface (7.) & the verification campaign (9.):", font="bold")],   
        [sg.Button('Calculate ratings', key="calculate_ratings", bind_return_key=True)],
        
        [sg.Text('−'*64)],
        
        [sg.Text("13. Submit ratings to microworkers:", font="bold")],   
        [sg.Radio('Use admin rating', 'loss', size=(12, 1), default=True, key="radio_admin"), sg.Radio('Use crowd rating', 'loss', size=(12, 1), key="radio_crowd")],
        
        [sg.Button('Submit ratings', key="submit_rating", bind_return_key=True)]             
    ]
    
    layout_final = [
        [
            sg.Column(layout_left),  # , element_justification="c"
            sg.VSeparator(),
            sg.Column(layout_right)
        ]
    ]

    window = sg.Window(title="Control Panel", layout=layout_final)  # , keep_on_top=True) #.Finalize()

    while True:
        event, values = window.read()

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        try:
            
            if event == "refresh_active_campaigns_questions":
                # Check for campaign ready to be rated (Only works for 1 active campaign at a time)
                status = "refresh_active_campaigns"
                mw_api = fl.MW_API(config["microworkers"]["api_key"], config["microworkers"]["api_url"])   # call class MW_API
                campaign_info, _, _ = fl.get_active_campaign(mw_api, method="questions", config=config)

                lines2display = []
                for line2append in campaign_info:
                    if len(line2append) > 50:
                        lines2display.append(line2append[:len(line2append)//2])
                        lines2display.append(line2append[len(line2append)//2:])
                    else:
                        lines2display.append(line2append)

                window["campaign_list_questions"].Update(values=lines2display)
            
            if event == "refresh_active_campaigns":
                # Check for campaign ready to be rated (Only works for 1 active campaign at a time)
                status = "refresh_active_campaigns"
                
                mw_api = fl.MW_API(config["microworkers"]["api_key"], config["microworkers"]["api_url"])   # call class MW_API
                campaign_info, _, _ = fl.get_active_campaign(mw_api, method="acquisitions", config=config)

                lines2display = []
                for line2append in campaign_info:
                    if len(line2append) > 50:
                        lines2display.append(line2append[:len(line2append)//2])
                        lines2display.append(line2append[len(line2append)//2:])
                    else:
                        lines2display.append(line2append)

                window["campaign_list"].Update(values=lines2display)
                
                # sg.Listbox(values=camp_setting_list, enable_events=False, size=(70, 4), key="campaign_settings")
                
            if event == "create_campaign_acqui":
                # Update config with newly set number of acquisitions
                status = "Load config"
                path_entered = values['path_config']                
                config = read_config(path_entered)
                
                fl.create_campaign(method="acqui", config=config)
            
            if event == "ftp_download_match":
                print("ftp download and match")
                # Fetch data from bplaced server to local file system
                method = "acquisitions"
                directory = fl.set_dir(method=method)
                fl.fetch_data(directory["rootDir_server"], directory["rootDir_local"], directory["subDir"], config)
                
                # Establish API connection
                mw_api = fl.MW_API(config["microworkers"]["api_key"], config["microworkers"]["api_url"])
                
                # Get info for active campaign
                _, campaign_running, campaign_paused = fl.get_active_campaign(mw_api, method, config)
                
                # Match bplaced with microworkers data  # Only for 1 active acquisition campaign, so either campaign_running or campaign_paused is empty
                if campaign_running:
                    campaign = campaign_running 
                elif campaign_paused: 
                    campaign = campaign_paused

                fl.match_data(campaign, mw_api, directory["rootDir_local"], directory["subDir"], method=method, save_country=True)

            if event == "saveConfigIntegration":
                # Save config with currently set values
                print("Save set config ...")
                
                path_entered = values['path_config']
                                
                config_integration = {
                    "DBSCAN-1": {
                        "minpts": int(values["dbscan_minPts"]),
                        "epsilon": int(values["dbscan_epsilon"])    # [pix]
                    },
                    "DBSCAN-2": {
                        "minpts": int(values["dbscan_weak_minPts"]),
                        "epsilon": int(values["dbscan_weak_epsilon"])    # [pix]
                    },
                    "minimal_length": float(values["min_axis_length"]), # minimal length of acquisition
                    "max_len_deviation": float(values["max_len_deviation"]),
                    "max_dist_2_integrated_line": float(values["max_dist_2_integrated_line"]),
                    "max_distance_correspondence": float(values["max_distance_correspondence"])
                }
                
                config_job = {
                    "number_of_acquis": int(values["number_of_acquis"])
                }
                
                write_config(path_entered, config_integration, config_job)
                config = read_config(path_entered)
                
            if event == "saveConfig":
                # Save config with currently set values
                print("Save set config ...")
                
                path_entered = values['path_config']
                                
                config_integration = {
                    "DBSCAN-1": {
                        "minpts": int(values["dbscan_minPts"]),
                        "epsilon": int(values["dbscan_epsilon"])    # [pix]
                    },
                    "DBSCAN-2": {
                        "minpts": int(values["dbscan_weak_minPts"]),
                        "epsilon": int(values["dbscan_weak_epsilon"])    # [pix]
                    },
                    "minimal_length": float(values["min_axis_length"]), # minimal length of acquisition
                    "max_len_deviation": float(values["max_len_deviation"]),
                    "max_dist_2_integrated_line": float(values["max_dist_2_integrated_line"]),
                    "max_distance_correspondence": float(values["max_distance_correspondence"])
                }
                
                config_job = {
                    "number_of_acquis": int(values["number_of_acquis"])
                }
                
                write_config( path_entered, config_integration, config_job )
                config = read_config( path_entered )

            if event == "loadConfig":
                print("Load config")
                path_entered = values['path_config']                
                config = read_config(path_entered)
                
                window["dbscan_minPts"].Update(config["integration"]["DBSCAN-1"]["minpts"])
                window["dbscan_epsilon"].Update(config["integration"]["DBSCAN-1"]["epsilon"])
                window["dbscan_weak_minPts"].Update(config["integration"]["DBSCAN-2"]["minpts"])
                window["dbscan_weak_epsilon"].Update(config["integration"]["DBSCAN-2"]["epsilon"])
                
                window["min_axis_length"].Update(config["integration"]["minimal_length"])
                window["max_len_deviation"].Update(config["integration"]["max_len_deviation"])
                window["max_dist_2_integrated_line"].Update(config["integration"]["max_dist_2_integrated_line"])
                window["max_distance_correspondence"].Update(config["integration"]["max_distance_correspondence"])

            if event == "start_integration":    
                try:            
                    fl.integrate_main(config)
                    print("Integration finished.")
                except:
                    print("Error in integrateMain")
                
                print("----------------------------------------------------------------")

            if event == "ftp_upload_txt":
                # Confirmation Popup
                event_pop, values_po = sg.Window("Confirm Upload to Bplaced", 
                                                 [[sg.T("Confirm overwrite of admin & crowd rating input&result files on bplaced.\nDONT if question campaign is already running, or finished. \n.txt for crowd are randomized and integration is based on the order.\nIf the order is lost (overwrite) questions can no longer be connected to the specific answer")], 
                                                  [sg.B("OK"), sg.B("Cancel")]], keep_on_top=True).read(close=True)
                if event_pop == "OK":   # "Cancel"
                    fl.upload_rating_files(config)

            if event == "generate_links":
                links = fl.generate_links(config)
                
                # Write to listbox
                window["listbox_links"].Update(values=links)
            
            if event == "listbox_links":                
                pyperclip.copy(values["listbox_links"][0])
                print("Copied to clipboard \"{}\"".format(pyperclip.paste()))
            
            if event == "open_browser":                
                links = window["listbox_links"].Values
                fl.open_browser(links)
            
            if event == "ftp_download_admin":
                fl.ftp_download_ratings(config, method="admin")
                
            if event == "create_campaign_questions":
                
                fl.create_campaign(method="questions", config=config)

            if event == "ftp_download_crowd":
                fl.ftp_download_ratings(config, method="crowd")

            if event == "calculate_ratings":
                try:
                    fl.calculate_ratings(config)
                except:
                    print("Error calculationg ratings")
            
            if event == "submit_rating":                
                if values["radio_admin"] and not values["radio_crowd"]:
                    rateMethod = "admin"
                elif values["radio_crowd"] and not values["radio_admin"]:
                    rateMethod = "crowd"
                
                print("Submit rating based on {} rating".format(rateMethod))
                fl.submit_rating(config, rateMethod)
            
        except:
            status = "Error executing event: \"{}\"".format(event)
        
        # window["campaign_list"].Update( values= [status,] )


if __name__ == "__main__":
    main_gui()
