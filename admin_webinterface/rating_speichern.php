<?php
    /* Retrieve data from AJAX POST request
    Input:
        "dataArray": JSON String
            Example structure of "dataArray":
            [
                {   
                    "clusterIdx":35,
                    "acquiIdc":[0,1,2],
                    "answer":"Yes",
                    "reference":{
                        "x":423.5,
                        "y":144,
                        "z":266.5,
                        "w":11
                    }
                },
                {   
                    "clusterIdx":36,
                    "acquiIdc":[3,4,5,6,7,8,9,10,11],
                    "answer":"No"
                }
            ]

    Return:
        .txt file with [Cluster Idx [idx] | Acquisition Idc [idx1 idx2 idx3] | Answer [Yes/No] | Reference in original image frame coordiante system [x y z w]]
    */

    // Retrieve and decode data
    $data = json_decode($_POST['dataArray'], true); // return as array
    $batch_numb = $_POST['batch_numb'];
    $cur_method = $_POST['method'];

    var_dump($data);
    //print $data[0]->{'clusterIdx'};
    //echo("<br><br>");
    //echo("batch_numb =". $batch_numb);
    //echo("<br><br>");
    //var_dump($data[0]);
    //echo("<br><br>");
    //var_dump(sizeof($data));
    //echo("data = ". $data[0]["answer"]);

    // Write to file
    //$file = fopen('Admininterface/Post Rating/'.$batch_numb.'_'.$cur_method.'.txt', 'w+');
    $file = fopen('Post Rating/'.$batch_numb.'_'.$cur_method.'.txt', 'w+');


    fwrite($file, "Rating Results\nCluster Idx [idx] | Acquisition Idc [idx1 idx2 idx3] | Answer [Yes/No] | Reference in HTML canvas coord system | Reference in original image frame coordiante system [x y z w]");
    
    for ($i = 0; $i < sizeof($data); $i++){
        $fwrite = fwrite($file, "\n");
        $fwrite = fwrite($file,  $data[$i]["clusterIdx"]);
        $fwrite = fwrite($file,  ",");
        for ( $j = 0; $j < sizeof($data[$i]["acquiIdc"]); $j++ ){
            $fwrite = fwrite($file,  $data[$i]["acquiIdc"][$j]);
            if ($j < sizeof($data[$i]["acquiIdc"])-1){
                $fwrite = fwrite($file,  " ");      
            }    
        }
        $fwrite = fwrite($file,  ",");
        $fwrite = fwrite($file,  $data[$i]["answer"]);
        if ( $data[$i]["answer"] == "Yes" ) {
            $fwrite = fwrite($file,  ",");            
            // Reference canvas coord
            $fwrite = fwrite($file,  $data[$i]["reference"]["x"]);
            $fwrite = fwrite($file,  " ");
            $fwrite = fwrite($file,  $data[$i]["reference"]["y"]);
            $fwrite = fwrite($file,  " ");
            $fwrite = fwrite($file,  $data[$i]["reference"]["z"]);
            $fwrite = fwrite($file,  " ");
            $fwrite = fwrite($file,  $data[$i]["reference"]["w"]);
            $fwrite = fwrite($file,  ",");
            // Reference original coord
            $fwrite = fwrite($file,  $data[$i]["referenceOri"]["x"]);
            $fwrite = fwrite($file,  " ");
            $fwrite = fwrite($file,  $data[$i]["referenceOri"]["y"]);
            $fwrite = fwrite($file,  " ");
            $fwrite = fwrite($file,  $data[$i]["referenceOri"]["z"]);
            $fwrite = fwrite($file,  " ");
            $fwrite = fwrite($file,  $data[$i]["referenceOri"]["w"]);
        }
    };
    fclose($file);    
?>
