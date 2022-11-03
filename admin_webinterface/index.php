<!DOCTYPE HTML>

<html>

<head>

  <meta charset="UTF-8">
  <title>Point Labelling</title>

  <script src="js/three.min.js"></script>
  <script src="js/OrbitControls.js"></script>
  <script src="js/PCDLoader.js"></script>

  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>

  <script>
    var lis;

    buffval = 500; // 680

    function plotoncanvas2D(img_src, target_context, target_canvas) {
      var img = new Image();
      img.onload = function() {
        var height = img.height;
        width = img.width;
        img.height = 500;
        target_canvas.width = buffval;
        target_canvas.height = img.height;
        target_context.drawImage(img, 0, 0, buffval, 500, 0, 0, target_canvas.width, target_canvas.height);

        // Original image size
        origWidth = img.naturalWidth;
        origHeight = img.naturalHeight;

        // Draw cluster center & acquisitions for first cluster
        initDrawing();

        // Restore already rated data if possible
        if (radio_answer_list[cur_cluster_idx]) {
          console.log("!! not empty -> restore answer selection !!")
          restore_answers();
          draw_reference();
          // Enable redraw button
          document.getElementById("redrawReferenceId").disabled = false;
        } else {
          // Disable redraw reference button
          document.getElementById("redrawReferenceId").disabled = true;
        }

        // Set toggle buttons
        document.getElementById("clusterDisplaySetting").innerHTML = "True";
        document.getElementById("clusterDisplaySetting").style.backgroundColor = "Green";
        document.getElementById("acquiDisplaySetting").innerHTML = "True";
        document.getElementById("acquiDisplaySetting").style.backgroundColor = "Green";        
        document.getElementById("referenceDisplaySetting").innerHTML = "True";
        document.getElementById("referenceDisplaySetting").style.backgroundColor = "Green";        
      }
      img.src = img_src
    }

    function restore_answers() {
      console.log("restore answers")
      let a1 = radio_answer_list[cur_cluster_idx].answer;

      if (a1 == "Yes") {
        document.getElementById("quest_q1_y").checked = true;
        // Enable redraw button
        document.getElementById("redrawReferenceId").disabled = false;
      } else if (a1 == "No") {
        document.getElementById("quest_q1_n").checked = true;
        // Disable redraw button until "Yes" is selected
        document.getElementById("redrawReferenceId").disabled = true;
      }
    }

    function uncheck_answers() {
      console.log("uncheck answers")
      // Reset Radio Buttons / uncheck
      var ele = document.getElementsByName("radio_q1");
      for (var i = 0; i < ele.length; i++) ele[i].checked = false;
    }

    function redrawReference() {            
      // Clear buffer
      contextbuffer.clearRect(0, 0, canvasbuffer.width, canvasbuffer.height);
      displayAll();

      updateCanvas();

      activate_draw();

    }

    function displayAll() {
      showReference = 1;
      showCluster = 1;
      showAcquisitions = 1;

      document.getElementById("clusterDisplaySetting").innerHTML = "True";
      document.getElementById("clusterDisplaySetting").style.backgroundColor = "Green";
      document.getElementById("acquiDisplaySetting").innerHTML = "True";
      document.getElementById("acquiDisplaySetting").style.backgroundColor = "Green";        
      document.getElementById("referenceDisplaySetting").innerHTML = "True";
      document.getElementById("referenceDisplaySetting").style.backgroundColor = "Green";
    }
    
    function toggleReference() {
      if ( showReference ) {
        showReference = 0; // Remove Reference
        document.getElementById("referenceDisplaySetting").innerHTML = "False";
        document.getElementById("referenceDisplaySetting").style.backgroundColor = "Red";
      } else {        
        showReference = 1; // Display Reference
        document.getElementById("referenceDisplaySetting").innerHTML = "True";
        document.getElementById("referenceDisplaySetting").style.backgroundColor = "Green";
      }
      updateCanvas();
    }

    function toggleAcquisitions() {
      if ( showAcquisitions ) {
        showAcquisitions = 0; // Remove acquistions
        document.getElementById("acquiDisplaySetting").innerHTML = "False";
        document.getElementById("acquiDisplaySetting").style.backgroundColor = "Red";
      } else {        
        showAcquisitions = 1; // Display acquisitions
        document.getElementById("acquiDisplaySetting").innerHTML = "True";
        document.getElementById("acquiDisplaySetting").style.backgroundColor = "Green";
      }
      updateCanvas();
    }

    function toggleCluster() {
      if ( showCluster ) {
        showCluster = 0; // Remove cluster center        
        document.getElementById("clusterDisplaySetting").innerHTML = "False";
        document.getElementById("clusterDisplaySetting").style.backgroundColor = "Red";
      } else {        
        showCluster = 1; // Display cluster center        
        document.getElementById("clusterDisplaySetting").innerHTML = "True";
        document.getElementById("clusterDisplaySetting").style.backgroundColor = "Green";
      }
      updateCanvas();
    }

    function activate_draw() {
      console.log( "Drawing activated" )
      drawActive = 1;
    }
    
    function drawcar() {
      console.log( " Draw car clicked " )
      if ( !drawActive ) {
        console.log( " drawing disabled ", drawActive )
        return;
      }

      // Check if chosen point is outside of viable image frame
      rect = canvasbuffer.getBoundingClientRect();
      mousemove.x = ((event.clientX - rect.left));
      mousemove.y = ((event.clientY - rect.top));
      // Show tooltip outside of the image
      let outsideX = mousemove.x > xRightEdge || mousemove.x < xLeftEdge
      let outsideY = mousemove.y > yBottomEdge || mousemove.y < yTopEdge
      if ( outsideX ) {
        let tooltipText = "Outside viable x";
        document.getElementById("logTextArea").innerHTML = tooltipText;
        console.log( "Pointer outside of image frame: x-axis")
        return;
      } else {
        document.getElementById("logTextArea").innerHTML = "";
      }
      if ( outsideY ) {
        let tooltipText = "Outside viable y";
        document.getElementById("logTextArea").innerHTML = tooltipText;
        console.log( "Pointer outside of image frame: y-axis")
        return;
      } else {
        document.getElementById("logTextArea").innerHTML = "";
      }

      // Draw point
      if ( init_point != 1 ) {    // init_point = 0 -> no starting point placed yet

        // Get current pointer location
        rect = canvasbuffer.getBoundingClientRect();
        mousestart.x = ((event.clientX - rect.left)) - 0.5;
        mousestart.y = Math.round((event.clientY - rect.top) - 0.61);

        //console.log(" rect -> ", rect)
        //console.log(" startPoint coordinates --> ", mousestart)

        // Store location
        cur_reference = { 
          x: mousestart.x, 
          y: mousestart.y
        };

        init_point = 1;
        // Disable buttons
        document.getElementById("redrawReferenceId").disabled = true;
        document.getElementById("toggleAcquisitionsId").disabled = true;
        document.getElementById("toggleClusterId").disabled = true;
        document.getElementById("toggleReferenceId").disabled = true;   
        document.getElementById("prevClusterId").disabled = true;     
        document.getElementById("nextClusterId").disabled = true;     
        document.getElementById("submit").disabled = true;     
        document.getElementById("quest_q1_y").disabled = true;      
        document.getElementById("quest_q1_n").disabled = true;
        
      } else {
		
        rect = canvasbuffer.getBoundingClientRect();
        mousestart.x = ((event.clientX - rect.left)) - 0.5;
        mousestart.y = Math.round((event.clientY - rect.top) - 0.61);

        //console.log(" rect -> ", rect)
        //console.log(" endPoint coordinates --> ", mousestart)

        init_point = 0;

        // Disable drawing features
        drawActive = 0;

        // Save reference coordinates
        cur_reference.z = mousestart.x
        cur_reference.w = mousestart.y
        
        radio_answer_list[ cur_cluster_idx ].reference = cur_reference;

        //console.log(cur_reference.x);console.log(cur_reference.y);console.log(cur_reference.z);console.log(cur_reference.w);

        // Transform reference coordinates to original image frame (to calculate distance later on)
        let cur_reference_ori = {};
        cur_reference_ori.x = (cur_reference.x / scale) + sx;
        cur_reference_ori.y = (cur_reference.y / scale) + sy;
        cur_reference_ori.z = (cur_reference.z / scale) + sx;
        cur_reference_ori.w = (cur_reference.w / scale) + sy;
                
        radio_answer_list[ cur_cluster_idx ].referenceOri = cur_reference_ori;
        //console.log("radio_answer_list->cur_reference", radio_answer_list[ cur_cluster_idx ].reference)
        //console.log("radio_answer_list->cur_reference_ori", radio_answer_list[ cur_cluster_idx ].referenceOri)
        
        // Draw acquired reference acquisition
        updateCanvas();
        
        // Enable buttons
        document.getElementById("redrawReferenceId").disabled = false;
        document.getElementById("toggleAcquisitionsId").disabled = false;
        document.getElementById("toggleClusterId").disabled = false;
        document.getElementById("toggleReferenceId").disabled = false;   
        document.getElementById("prevClusterId").disabled = false;     
        document.getElementById("nextClusterId").disabled = false;     
        document.getElementById("submit").disabled = false;     
        document.getElementById("quest_q1_y").disabled = false;      
        document.getElementById("quest_q1_n").disabled = false;          
      }
    }

    function drawX(x, y) {
      contextbuffer.setLineDash([5, 3]);
      contextbuffer.beginPath();
      // contextbuffer.moveTo(x, y - 10);
      // contextbuffer.lineTo(x, y + 10);
      // contextbuffer.moveTo(x + 10, y);
      // contextbuffer.lineTo(x - 10, y);
      contextbuffer.stroke();
    }

    function preline() {
      //console.log(" preline event started")

      // Stop rotine, when drawing disabled
      if ( !drawActive ) {
        //console.log( " drawing disabled" )
        return;
      }
      
      // Plot line between 1st point and current mouse position 
      if ( init_point ) { 
        // Clear buffer and get mouseposition
        contextbuffer.clearRect(0, 0, canvasbuffer.width, canvasbuffer.height);
        rect = canvasbuffer.getBoundingClientRect();
        mousemove.x = ((event.clientX - rect.left));
        mousemove.y = ((event.clientY - rect.top));
        //console.log("PX",mousemove.x)
        //console.log("PY",mousemove.y)
        contextbuffer.strokeStyle = 'magenta';

        // Plot line
        //let cur_reference = reference_list[ cur_cluster_idx ]
        //console.log(cur_reference)
        contextbuffer.beginPath();
        contextbuffer.arc( cur_reference.x, cur_reference.y, 2, 0, 2 * Math.PI );
        drawX( cur_reference.x, cur_reference.y );
        contextbuffer.lineWidth = 1;

        contextbuffer.stroke();
        contextbuffer.beginPath();
        drawX( mousemove.x, mousemove.y );

        contextbuffer.moveTo( cur_reference.x, cur_reference.y );
        contextbuffer.lineTo( mousemove.x, mousemove.y );
        contextbuffer.lineWidth = linestrength;
        contextbuffer.setLineDash([15, 5]);
        contextbuffer.stroke();

        // Show tooltip outside of the image
        let outsideX = mousemove.x > xRightEdge || mousemove.x < xLeftEdge
        let outsideY = mousemove.y > yBottomEdge || mousemove.y < yTopEdge

        if ( outsideX ) {
          let tooltipText = "Caution: Outside viable x -> Choose Lineend inside valid image frame"
          document.getElementById("logTextArea").innerHTML = tooltipText;
          console.log( "outside of image bounding box: x-axis")
        } else {
          document.getElementById("logTextArea").innerHTML = "";
        }
        if ( outsideY ) {
          let tooltipText = "Caution: Outside viable y -> Choose Lineend inside valid image frame"
          document.getElementById("logTextArea").innerHTML = tooltipText;
          console.log( "outside of image bounding box: y-axis")
        } else {
          document.getElementById("logTextArea").innerHTML = "";
        }
      }
    }

    function drawAcquisitions( first_idx, last_idx ) {
      // Acquisitions for current cluster      
      if ( !showAcquisitions ) {
        return;
      }

      for ( i=first_idx; i<last_idx+1; i++ ) {
        cur_acqui = cur_acqui_list[i];   
        //console.log( " cur_acqui --> ", cur_acqui_list[ i ] )

        // Draw acquisition
        // Trafo: source coord -> image detail coord -> destination canvas coord
        var xSource_1 = cur_acqui.x1;
        var xSource_2 = cur_acqui.x2;
        var ySource_1 = cur_acqui.y1;
        var ySource_2 = cur_acqui.y2;

        var xSection_1 = xSource_1 - sx;
        var xSection_2 = xSource_2 - sx;
        var ySection_1 = ySource_1 - sy;
        var ySection_2 = ySource_2 - sy;

        var xDestination_1 = xSection_1 * scale; //cur_acqui.x + x_off;
        var xDestination_2 = xSection_2 * scale;
        var yDestination_1 = ySection_1 * scale; //cur_acqui.x + x_off;
        var yDestination_2 = ySection_2 * scale;

        contextbuffer.lineWidth = 3; //linestrength;
        contextbuffer.strokeStyle = 'red';
        contextbuffer.setLineDash([15, 0]);
        contextbuffer.beginPath();
        contextbuffer.moveTo(xDestination_1, yDestination_1);
        contextbuffer.lineTo(xDestination_2, yDestination_2);
        contextbuffer.stroke();
      }
    }

    function draw_reference() {      
      console.log("drawing reference")
      if ( !showReference ) {
        return;
      }

      if ( radio_answer_list[ cur_cluster_idx ] === undefined ) {
        console.log( "No reference for current cluster found")
        return;
      }

      cur_reference = radio_answer_list[ cur_cluster_idx ].reference;
      if ( cur_reference ) {
        contextbuffer.lineWidth = 3; //linestrength;
        contextbuffer.strokeStyle = 'yellow';
        contextbuffer.setLineDash([15, 0]);
        contextbuffer.beginPath();
        contextbuffer.moveTo(cur_reference.x, cur_reference.y);
        contextbuffer.lineTo(cur_reference.z, cur_reference.w);
        contextbuffer.stroke();
      }
    }

    function drawCluster() {
      if ( !showCluster ) {
        return;
      }

      // Set draw style
      contextbuffer.lineWidth = 3; //linestrength;
      contextbuffer.strokeStyle = 'blue'; //"rgba(0, 0, 255, 0.5)"; //'magenta';
      contextbuffer.setLineDash([15, 0]);
      contextbuffer.beginPath();
      
      // Trafo: source coord -> image detail coord -> destination canvas coord
      //scale = canvasshad.width / sWidth; // ratio width/height = 1

      // Calculate new cluster location -> middle of destination canvas
      var xCluster = gapToEdge; //xSource - sx;
      var yCluster = gapToEdge; //ySource - sy;

      var xDestination = xCluster * scale;  // == 250
      var yDestination = yCluster * scale;  // == 250
      //console.log("xDestination, yDestination", xDestination, yDestination)

      // Draw cluster center
      contextbuffer.arc(xDestination, yDestination, 10, 0, 2 * Math.PI); // draw circle
      contextbuffer.stroke();
    }

    function setupImage(){
      //console.log(sx, sy)
      var img = new Image();
      img.onload = function() {
        canvasshad.width = 500;
        canvasshad.height = 500; //img.height;
        contextshad.drawImage(img, sx, sy, sWidth, sHeight, 0, 0, canvasshad.width, canvasshad.height);
      }
      img.src = img_src
    }

    function initDrawing() {
      // Data to display -> sinnlos, kann eigentlich ersetzt werden
      //cur_acqui_list = uncertain_cluster_data_list2; //console.log( "cur_acqui_list", cur_acqui_list )
      cur_acqui_list = uncertain_data; //uncertain_cluster_data_list_ell
      
      // Clear the entire canvas
      contextbuffer.clearRect(0, 0, canvasbuffer.width, canvasbuffer.height);
      contextghost.clearRect(0, 0, canvasbuffer.width, canvasbuffer.height);      

      // Get cluster ids
      idc_all_cluster = cur_acqui_list.map(function(o) { return o.cluster_idx; })
      //console.log("idc of all cluster ->", idc_all_cluster)

      // Get unique cluster ids
      const distinct = (value, index, self) => {
        return self.indexOf(value) === index;
      }
      uniqueIds = idc_all_cluster.filter( distinct );
      //console.log("uniqueIds --> ", uniqueIds)

      // Display first entry
      // Get index
      idc_cur_cluster = [];
      idc_all_cluster.filter( function( elem, index, array ){
          if( elem == uniqueIds[0] ) {
            idc_cur_cluster.push( index );
          }
      });
      //console.log( "indices of first cluster --> ", idc_cur_cluster );
      //console.log( "first and last idc", idc_cur_cluster[0], idc_cur_cluster)

      // Construct image, using first cluster
      let xCluster = cur_acqui_list[0].xCluster; // Cluster center
      let yCluster = cur_acqui_list[0].yCluster;
      sx = xCluster - gapToEdge;
      sy = yCluster - gapToEdge;
      //console.log( "xCluster, yCluster", xCluster, yCluster )
      
      // Update batch/job label
      document.getElementById("jobLabel").innerHTML = cur_batch;

      // Update the progress label
      updateProgress();

      // Update image frame info
      boundingBox( xCluster, yCluster );

      // Setup the image
      setupImage();

      // Draw cluster center
      drawCluster();

      // Draw acquisitions
      drawAcquisitions( idc_cur_cluster[0], idc_cur_cluster[ idc_cur_cluster.length -1 ] );

      // Enable 
      enableRadioButton();
    }

    function enableRadioButton() {
      document.getElementById("quest_q1_y").disabled = false;
      document.getElementById("quest_q1_n").disabled = false;

      // Add listener
      document.getElementsByName("radio_q1").forEach( (elem) => {
        elem.addEventListener("change", function(event) {
          var item = event.target.value;
          console.log("RADIO BUTTON CHANGE", item);
          if ( item == "Yes" ) {
            // Start reference drawing
            activate_draw();

            // Update idc
            idc_cur_cluster = [];
            idc_all_cluster.filter( function( elem, index, array ){
                if( elem == uniqueIds[ cur_cluster_idx ] ) {
                  idc_cur_cluster.push( index );
                }
            });
            // Save data
            radio_answer_list[ cur_cluster_idx ] = {
              clusterIdx: uniqueIds[ cur_cluster_idx ],
              acquiIdc: idc_cur_cluster,
              answer: item
            };
            // Display task
            let tooltipText = "Choose starting point of reference acquisition"
            document.getElementById("logTextArea").innerHTML = tooltipText;
            // Enable redraw button
            document.getElementById("redrawReferenceId").disabled = false;
            //console.log("radio_answer_list[ cur_cluster_idx ]", radio_answer_list[ cur_cluster_idx ])
          } else {
            // Load next cluster
            let r = confirm("Rate next cluster")
            if ( r == true ){
              // Update idc
              idc_cur_cluster = [];
              idc_all_cluster.filter( function( elem, index, array ){
                  if( elem == uniqueIds[ cur_cluster_idx ] ) {
                    idc_cur_cluster.push( index );
                  }
              });
              // Save data        
              radio_answer_list[ cur_cluster_idx ] = {
                clusterIdx: uniqueIds[ cur_cluster_idx ],
                acquiIdc: idc_cur_cluster,
                answer: item
              };
              //console.log("radio_answer_list[ cur_cluster_idx ]", radio_answer_list[ cur_cluster_idx ])
              nextCluster();
            }
          }
        });
      });
    }

    function updateCanvas() {      
      console.log("--update canvas clicked--")
      // Clear the entire canvas  
      contextbuffer.clearRect(0, 0, canvasbuffer.width, canvasbuffer.height);
      contextghost.clearRect(0, 0, canvasbuffer.width, canvasbuffer.height);

      // Get index
      idc_cur_cluster = [];
      idc_all_cluster.filter( function( elem, index, array ){
          if( elem == uniqueIds[ cur_cluster_idx ] ) {
            idc_cur_cluster.push( index );
          }
      });
      //console.log( " Current cluster --> ", cur_cluster_idx)
      //console.log( " Acquisition indices of cluster --> ", idc_cur_cluster );

      // Construct image, using current cluster
      let firstIdx = idc_cur_cluster[0]
      let lastIdx = idc_cur_cluster[ idc_cur_cluster.length-1 ]
      let xCluster = cur_acqui_list[ firstIdx ].xCluster // Cluster center
      let yCluster = cur_acqui_list[ firstIdx ].yCluster
      sx = xCluster - gapToEdge;
      sy = yCluster - gapToEdge;
      console.log( "xCluster, yCluster", xCluster, yCluster )

      // Update image frame info
      boundingBox( xCluster, yCluster )

      // Setup image      
      setupImage();

      // Draw cluster center
      if ( showCluster ) {
        console.log( "Draw cluster: showCluster=", showCluster )
        drawCluster()
      } else {
        console.log( "Skip cluster drawing: showCluster=", showCluster )
      }      

      // Draw acquisitions
      if ( showAcquisitions ) {
        console.log( "Draw acquisitions: showAcquisitions=", showAcquisitions )
        drawAcquisitions( firstIdx, lastIdx );
      } else {
        console.log( "Skip acquistion drawing: showAcquisitions", showAcquisitions )
      }
      
      // Draw reference
      if ( showReference ) {
        console.log( "Draw reference: showReference=", showReference )
        draw_reference();
      } else {
        console.log( "Skip reference drawing: showReference", showReference )
      }
    }

    function boundingBox( xCluster, yCluster ) {
      // Caculate edges of bounding box for specific cluster to prevent drawing outside of the zoomed in image
      // Original image size
      //console.log("origWidth, origHeight", origWidth, origHeight)        
      
      // Bounding box of original image
      let xMin = 0; let xMax = origWidth;
      let yMin = 0; let yMax = origHeight; 
      //console.log("xMin, xMax, yMin, yMax", xMin, xMax, yMin, yMax)

      // Calculate possible edges for current cluster 
      if ( (xCluster - gapToEdge) < xMin ) {
        xLeftEdge = xMin;          
      } else { xLeftEdge = xCluster - gapToEdge };

      if ( (xCluster + gapToEdge) > xMax) {
        xRightEdge = xMax;
      } else { xRightEdge = xCluster + gapToEdge };

      if ( (yCluster - gapToEdge) < yMin) {
        yTopEdge = yMin;
      } else { yTopEdge = yCluster - gapToEdge };

      if ( (yCluster + gapToEdge) > yMax ) {
        yBottomEdge = yMax;
      } else { yBottomEdge = yCluster + gapToEdge };
      
      // Transform to destination image coordinates
      scale = canvasshad.width / sWidth; // ratio width/height = 1
      xLeftEdge = ( xLeftEdge - sx ) * scale;
      xRightEdge = ( xRightEdge - sx ) * scale;
      yTopEdge = ( yTopEdge - sy ) * scale;
      yBottomEdge = ( yBottomEdge - sy ) * scale;

      //console.log("scale", scale)
      //console.log("xLeftEdge, xRightEdge", xLeftEdge, xRightEdge)
      //console.log("yTopEdge, yBottomEdge", yTopEdge, yBottomEdge)
    }

    function updateProgress() {
      let cur_cluster_idx_ = cur_cluster_idx +1;  // idc = [1 2 3 4 ...]
      let percent = cur_cluster_idx / uniqueIds.length * 100;
      document.getElementById("progressLabel").innerHTML = cur_cluster_idx_ + "/" + uniqueIds.length; //+ "<br>" + percent + "% of Total Cluster";
    }
    
    function nextCluster() {
      console.log("--next center clicked--")

      // Confirm radio selection
      const radio_q1 = document.querySelectorAll('input[name="radio_q1"]');
      let radio_a1
      for (const rb of radio_q1) {
        if (rb.checked) {
          radio_a1 = rb.value;
          break;
        }
      }
      if (typeof radio_a1 === 'undefined') {
        alert("Please answer the question and decide if its a cluster or not")
        return;
      }
      console.log("radio_a1", radio_a1)
      cur_reference = radio_answer_list[ cur_cluster_idx ].reference;
      if ( (radio_a1 === "Yes") && (!cur_reference) ) {
        alert("Mark the reference, or choose 'no' as answer");
        return;
      }

      // Display Acquisitions, cluster and reference
      displayAll();

      //console.log(" cur_acqui_list.length --> ", cur_acqui_list.length ) 
      if ( cur_acqui_list.length == 0 ) {
        alert("ERROR acquistion list empty")        
        return;
      }        

      // Clear the entire canvas  
      contextbuffer.clearRect(0, 0, canvasbuffer.width, canvasbuffer.height);
      contextghost.clearRect(0, 0, canvasbuffer.width, canvasbuffer.height);

      // Get current cluster idx
      cur_cluster_idx = cur_cluster_idx + 1;
      if (cur_cluster_idx > uniqueIds.length - 1) { // show last entry
        cur_cluster_idx -= 1;
        // Display msg
        let tooltipText = "Last cluster reached, save acquired ratings or control recent references"
        document.getElementById("logTextArea").innerHTML = tooltipText;
      } else {
        //prog = prog + progstep; // Update Progressbar    
      }

      // Update progress
      updateProgress();

      // Restore Selected Answers if possible
      if ( radio_answer_list[ cur_cluster_idx ] ) {
        restore_answers();
      } else {
        uncheck_answers();
      }

      // Get index
      idc_cur_cluster = [];
      idc_all_cluster.filter( function( elem, index, array ){
          if( elem == uniqueIds[ cur_cluster_idx ] ) {
            idc_cur_cluster.push( index );
          }
      });
      //console.log( " Current cluster --> ", cur_cluster_idx)
      //console.log( " Acquisition indices of cluster --> ", idc_cur_cluster );

      // Construct image, using current cluster
      let firstIdx = idc_cur_cluster[0];      
      let lastIdx = idc_cur_cluster[ idc_cur_cluster.length-1 ];
      let xCluster = cur_acqui_list[firstIdx].xCluster; // Cluster center
      let yCluster = cur_acqui_list[firstIdx].yCluster;
      sx = xCluster - gapToEdge;
      sy = yCluster - gapToEdge;

      boundingBox( xCluster, yCluster );

      setupImage();

      drawCluster();
  
      drawAcquisitions( firstIdx, lastIdx );
      
      draw_reference();      
    }

    function previousCluster() {      
      console.log("--previousCluster clicked--")    
      // Display Acquisitions, Cluster and reference
      displayAll();
      
      //console.log(" cur_acqui_list.length --> ", cur_acqui_list.length ) 
      if ( cur_acqui_list.length == 0 ) {
        console.log("--switch to uncertain cluster--")
        switch_to_uncertain_cluster();
        return;
      }

      // Clear the entire canvas  
      contextbuffer.clearRect(0, 0, canvasbuffer.width, canvasbuffer.height);
      contextghost.clearRect(0, 0, canvasbuffer.width, canvasbuffer.height);

      //var progstep = 100 / cur_acqui_list.length

      cur_cluster_idx = cur_cluster_idx - 1;
      //console.log("cur_cluster_idx", cur_cluster_idx)
      if (cur_cluster_idx < 0) { // show first entry
        cur_cluster_idx = 0;
        // Display msg
        let tooltipText = "First cluster reached, rate/control acquired/remaining cluster or save acquired ratings"
        document.getElementById("logTextArea").innerHTML = tooltipText;
        //prog = -50;                 // Update Progressbar  
      } else {
        //prog = prog - progstep; // Update Progressbar     
      }

      updateProgress();

      // Restore Selected Answers if possible
      if (radio_answer_list[cur_cluster_idx]) {
        console.log("!! not empty -> restore answer selection !!")
        restore_answers();
      } else {
        uncheck_answers();
      }

      // Get index
      var idc_cur_cluster = [];

      idc_all_cluster.filter( function( elem, index, array ){
          if( elem == uniqueIds[ cur_cluster_idx ] ) {
            idc_cur_cluster.push( index );
          }
      });
      console.log( " Current cluster --> ", cur_cluster_idx)
      console.log( " Acquisition indices of cluster --> ", idc_cur_cluster );

      // Construct image, using current cluster
      let firstIdx = idc_cur_cluster[0]
      let lastIdx = idc_cur_cluster[ idc_cur_cluster.length-1 ]
      let xCluster = cur_acqui_list[firstIdx].xCluster // Cluster center
      let yCluster = cur_acqui_list[firstIdx].yCluster
      sx = xCluster - gapToEdge;
      sy = yCluster - gapToEdge;

      boundingBox( xCluster, yCluster );

      setupImage();
      
      drawCluster();

      drawAcquisitions( firstIdx, lastIdx );

      draw_reference();
    }

    function saveRatings() {
      /*
      Write rating data to .txt File
      */
      console.log( radio_answer_list[0].referenceOri )
      var ratingDataJSON = JSON.stringify( radio_answer_list );
      console.log("JSON ", ratingDataJSON)
      $.ajax({
        type: "POST",
        url: "rating_speichern.php",
        data: { 
          dataArray : ratingDataJSON,
          batch_numb : cur_batch,
          method: cur_method
         },
        success: function() {
          alert("Rating submitted to rating_speichern.php");
        } 
      });
    }
  </script>

</head>
<!-- END OF HEAD, START OF BODY /////////////////////////////////////////////-->

<body id="main_body">
  <div id="frame_color" style="background: #EEEEEE; width: 1100px ">
    <div id="frame_all" style="background: #DDDDDD; width: 1000px ">
      <?php
        // Parse URL -> batchIdx: specifies .shd file
        if (isset($_GET['batchIdx']) and isset($_GET['method'])) {
          $cur_batch = $_GET["batchIdx"];            // http://localhost/Admin_webinterface/?batchIdx=1&  method=ell or method=db
          $cur_method = $_GET["method"];
        } else {
          $cur_batch = 1000;  // Error code
          $cur_method = 1337;
        }
      ?>

      <div style="padding-left:30px; padding-right:30px; padding-top:20px; padding-bottom:50px; background: white;">
        <div id="Survey" class="tabcontent" style="background: white;">
          <table border="1" style="width:100%">
            <tr>
              <td style="width:80%;text-align:left;">
                <form name=F1>
                  <center>
                    <canvas id="placeholder" width="500" height="500"></canvas>

                    <div style="position: relative;">
                      <!-- width original 680 -->
                      <canvas id="layer1" width="500" height="500" style="box-shadow: 4px 4px 14px #000; position: absolute; left: 100px; top: -500px; z-index: 0; background:transparent;"></canvas>
                      <canvas id="layer_ghost" width="500" height="500" style="box-shadow: 4px 4px 14px #000; position: absolute; left: 100px; top: -500px; z-index: 1; background:transparent; "></canvas>
                      <canvas id="layer2" width="500" height="500" style="box-shadow: 4px 4px 14px #000; position: absolute; left: 100px; top: -500px; z-index: 2; background:transparent;" onmousemove="preline(event)" onclick="drawcar(event)"></canvas>
                    </div>
              </td>
              
              <script>
                // canvas element in DOM
                var canvasshad = document.getElementById('layer1');
                var contextshad = canvasshad.getContext('2d');

                // buffer canvas
                var canvasbuffer = document.getElementById('layer2');
                var contextbuffer = canvasbuffer.getContext('2d');

                var canvasghost = document.getElementById('layer_ghost');
                var contextghost = canvasghost.getContext('2d');

                // Drawing controls
                var mousestart = new THREE.Vector2();
                var mousemove = new THREE.Vector2();
                var mousecur = new THREE.Vector2();
                var referenceAcqui = [];

                var drawActive = 0;
                var showAcquisitions = 1; // Boolean, 1: acquistions
                var showReference = 1; // Boolean, 1: reference
                var showCluster = 1; // Boolean, 1: cluster centers
                var toggleUncertain = 1;

                var cur_reference = [];

                var lineStart = 0;
                var lineEnd = 0;

                var init_point = 0;

                var cluster_center_list = [];
                var curr_cluster_idx = 0;

                // Image config
                var origWidth; var origHeight // Original image size
                var sWidth = 300; var sHeight = 300; // Zoomed in area size                
                var gapToEdge = sWidth / 2; // gap to the edge from zoomed in center
                var xLeftEdge; var xRightEdge; var yBottomEdge; var yTopEdge; // Bounding box in destination canvas for zoomed in view
                var sx; var sy; var scale;

                // Path to cluster for current batch / job
                var cluster_src_2ndDBSCAN;
                var cluster_src_ell;

                // Lists to store textfile database
                var uncertain_data = [];
                var existing_rating_data = [];

                var uncertain_cluster_data_list = [];
                var uncertain_cluster_data_list2 = [];
                var uncertain_cluster_data_list_ell = [];
                var axis_too_short_data_list = [];
                var db_outlier_data_list = [];
                var db_integrated_outlier_data_list = [];
                var kmeans_outlier_data_list = [];
                var integrated_data_list = [];

                var radio_answer_list = [];
                var finalResults = [];

                var idc_cur_cluster = [];
                var cur_acqui_list = [];
                var cur_cluster_idx = 0;
                var idc_all_cluster = [];
                var uniqueIds = [];

                var label_id = 1;
                var linestrength = 1;
                var img_src;
                var width = 0;

                // Init first cluster with corresponding batch shd

                function loadData( srcPath ) {
                  var request = new XMLHttpRequest();
                    request.open("GET", srcPath);
                    request.addEventListener('load', function(event) {
                      if (request.status >= 200 && request.status < 300) { //https://wiki.selfhtml.org/wiki/JavaScript/XMLHttpRequest
                        // Format response
                        var data = request.responseText;
                        if (data != undefined){
                          data = data.split("\n");   // Split textblock to array
                          
                          if ( data[-1] == undefined ) {  // remove last entry if empty/undefined (happens when ftplib upload to server)
                            data.splice(-1) 
                          }

                          data.splice(0,2); // remove header
                          data.forEach( splitterFunctionOutlier ); 

                          uncertain_data = data; 

                          //console.log("uncertain_data", uncertain_data)

                          function splitterFunctionOutlier(item, index, arr) {
                            if (item == "") {
                              return;
                            }
                            splittedData = item.split(",")
                            //console.log("splittedData", splittedData)

                            coordAcqui = splittedData[1].split(" ");
                            coordCluster = splittedData[2].split(" ");
                            workerId = splittedData[3];

                            arr[index] = {
                              cluster_idx: parseInt(splittedData[0]),
                              x1: parseFloat(coordAcqui[0]),       
                              y1: parseFloat(coordAcqui[1]),
                              x2: parseFloat(coordAcqui[2]),
                              y2: parseFloat(coordAcqui[3]),
                              xCluster: parseFloat(coordCluster[0]),  // Cluster center
                              yCluster: parseFloat(coordCluster[1]),
                              workerId: String(workerId)
                            };
                          }
                        } 
                      } else {
                        console.warn(request.statusText, request.responseText);
                      }
                    });
                    request.send();
                }

                function loadRating( srcPath ) {
                  var request = new XMLHttpRequest();
                    request.open("GET", srcPath);
                    request.addEventListener('load', function(event) {
                      if (request.status >= 200 && request.status < 300) { //https://wiki.selfhtml.org/wiki/JavaScript/XMLHttpRequest
                        // Format response
                        var data = request.responseText;
                        existing_rating_data = data.split("\n"); // Split textblock to array
                        existing_rating_data.splice(0, 2); // remove header

                        //console.log("existing_rating_data", existing_rating_data)

                        existing_rating_data.forEach( splitterFunctionOutlier )

                        function splitterFunctionOutlier(item, index, arr) {
                          splittedData = item.split(",")
                          acquiIdc = splittedData[1].split(" ");
                          acquiIdc.map(function(o) {return parseInt(o, 10)});
                          
                          console.log("splittedData", splittedData)
                          console.log("splittedData.length", splittedData.length)
                          if ( splittedData.length < 4 ) {
                            arr[index] = {
                              clusterIdx: parseInt(splittedData[0]),
                              acquiIdc: acquiIdc,
                              answer: String(splittedData[2]),
                              reference: {},
                              referenceOri: {}
                            }
                          } else {
                            canvasCoordReference = splittedData[3].split(" ");
                            origCoordReference = splittedData[4].split(" ");
                            //console.log("acquiIdc", acquiIdc)
                            arr[index] = {
                              clusterIdx: parseInt(splittedData[0]),
                              acquiIdc: acquiIdc,
                              answer: String(splittedData[2]),
                              reference: {
                                x: parseFloat(canvasCoordReference[0]),
                                y: parseFloat(canvasCoordReference[1]),
                                z: parseFloat(canvasCoordReference[2]),
                                w: parseFloat(canvasCoordReference[3])                              
                              },
                              referenceOri: {
                                x: parseFloat(origCoordReference[0]),
                                y: parseFloat(origCoordReference[1]),
                                z: parseFloat(origCoordReference[2]),
                                w: parseFloat(origCoordReference[3])
                              }                            
                            };
                          };                         
                        }
                        // Update rating variable
                        radio_answer_list = existing_rating_data;
                      } else {
                        console.warn(request.statusText, request.responseText);
                      }
                    });
                    request.send();
                }

                // Get URL Parameters 
                var cur_batch = <?php echo $cur_batch ?>;
                
                if (( cur_batch != 1000 ) || (cur_method != 1337)) {
                  var cur_method = "<?php echo $cur_method ?>";
                  console.log("Data/job" + (cur_batch) + "/shd.png")
                  img_src = "Data/job" + (cur_batch) + "/shd.png";

                  //var rating_src = "Admininterface/Post Rating/" + cur_batch + "_" + cur_method + ".txt";
                  var rating_src = "Post Rating/" + cur_batch + "_" + cur_method + ".txt";

                  if (cur_method == "ell") {
                    // Load uncertain cluster
                    //cluster_src_ell = "Admininterface/Pre Rating/" + (cur_batch-1) + "_uncertain_cluster_ellipse.txt";        // -1 -> python jobs 0,1,2,....
                    cluster_src_ell = "Pre Rating/" + (cur_batch-1) + "_uncertain_cluster_ellipse.txt";
                    loadData( cluster_src_ell );

                    // Load existing ratings if already partly rated 
                    loadRating( rating_src );

                  } else if (cur_method == "db") {
                    //cluster_src_2ndDBSCAN = "Admininterface/Pre Rating/" + (cur_batch-1) + "_uncertain_cluster_db_weak.txt";      // -1 -> python jobs 0,1,2,....
                    cluster_src_2ndDBSCAN = "Pre Rating/" + (cur_batch-1) + "_uncertain_cluster_db_weak.txt";
                    loadData( cluster_src_2ndDBSCAN );

                    // Load existing ratings if already partly rated 
                    //rating_src_db = "Admininterface/Post Rating/" + cur_batch + "_" + cur_method + ".txt";
                    rating_src_db = "Post Rating/" + cur_batch + "_" + cur_method + ".txt";
                    loadRating( rating_src );
                  }

                  plotoncanvas2D("Data/job" + cur_batch + "/shd.png", contextshad, canvasshad); // +0 -> js jobs 1,2,....
                }

                //var cur_batch = <?php //echo $next_batch ?>;
                //var cur_it = <?php //echo $next_it ?>;

                console.log(cur_batch)
                //console.log(cur_it)

                var sos = new Date();
              </script>

              <td style="width:50%;text-align:center">                
                <p id="qTxt">Does the Cluster clearly mark a vehicle?</p>
                <input type="radio" name="radio_q1" id="quest_q1_y" value="Yes"> Yes
                <input type="radio" name="radio_q1" id="quest_q1_n" value="No"> No
                <br>
                <table style="width:100%">
                  <tr style="width:100%;text-align:left">
                    <td>Settings:</td>
                  </tr>
                  <tr>
                    <td style="text-align:left">Acquisitions</td>
                    <td id="acquiDisplaySetting" style="width:15mm">True</td>
                    <td><button id="toggleAcquisitionsId" type="button" onclick="toggleAcquisitions()">Toggle</button></td>
                  </tr>
                  <tr>
                    <td style="text-align:left">Cluster</td>
                    <td id="clusterDisplaySetting" style="width:15mm">True</td>
                    <td><button id="toggleClusterId" type="button" onclick="toggleCluster()">Toggle</button></td>
                  </tr>
                  <tr>
                    <td style="text-align:left">Reference</td>
                    <td id="referenceDisplaySetting" style="width:15mm">True</td>
                    <td><button id="toggleReferenceId" type="button" onclick="toggleReference()">Toggle</button></td>
                  </tr>
                </table>
                <br>
                <button id="redrawReferenceId" type="button" onclick="redrawReference()">Redraw Reference</button>
                <br><br>
                <table style="text-align:center">
                  <tr>
                    <td style="text-indent:35px;width:70%;text-align:left">Batch/Job:</td>
                    <td id="jobLabel" style="width:30%;text-align:center">placeholder</td>
                  </tr>
                  <tr>
                    <td  style="text-indent:35px;width:70%;text-align:left">Progress:</td>
                    <td id="progressLabel" style="width:30%;text-align:center">42/42</td>
                  </tr>
                </table>

                <button id="prevClusterId" type='button' onclick="previousCluster()" style=" width: 80px; font-size: 20px; font-weight:bold"> &lsaquo; </button>
                <button id="nextClusterId" type='button' onclick="nextCluster()" style=" width: 80px; font-size: 20px; font-weight:bold"> &rsaquo; </button>
                <br><br>
                <input ID="submit" type="button" name="submit" value="Save Ratings" onclick="saveRatings()" style=" width: 92%; font-size: 18px">
                <br><br><br>
                <p style="text-align:left">&nbsp;Log messages:</p>
                <textarea id="logTextArea" cols="num" rows="num" style="width:80%;height:100px;resize:none" readonly></textarea>
              </td>   
            </tr>       
          </table>                        
        </div>
      </div>

      <script>
        var body = document.body;
      </script>
    </div>
  </div>
</body>
</html>