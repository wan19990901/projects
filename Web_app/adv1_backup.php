<?php

/* Include the `fusioncharts.php` file that contains functions    to embed the charts. */

   include("fusioncharts.php");

/* The following 4 code lines contain the database connection information. Alternatively, you can move these code lines to a separate file and include the file here. You can also modify this code based on your database connection. */

   require_once "dbconf.php";

   // Establish a connection to the database
   $dbhandle = new mysqli($host, $username, $password, $dbname);

   /*Render an error message, to avoid abrupt failure, if the database connection parameters are incorrect */
   if ($dbhandle->connect_error) {
      exit("There was an error with your connection: ".$dbhandle->connect_error);
   }
?>

<html>
   <head>
      <title>Advanced Function 1</title>
    <link  rel="stylesheet" type="text/css" href="css/style.css" />

      <!-- You need to include the following JS file to render the chart.
      When you make your own charts, make sure that the path to this JS file is correct.
      Else, you will get JavaScript errors. -->

      <script src="js/fusioncharts.js"></script>
  </head>

   <body>
     <center>
      <?php

         // Form the SQL query that returns the top 10 most populous countries
         $strQuery ="select gname,scores from adv1";

         // Execute the query, or else return the error message.
         $result = $dbhandle->query($strQuery) or exit("Error code ({$dbhandle->errno}): {$dbhandle->error}");

         // If the query returns a valid response, prepare the JSON string
         if ($result) {
            // The `$arrData` array holds the chart attributes and data
            $arrData = array(
                "chart" => array(
                  "caption" => "Top 10 Most Dangerous Gang",
                  "showValues" => "0",
                  "theme" => "zune"
                  )
               );

            $arrData["data"] = array();

    // Push the data into the array
            while($row = mysqli_fetch_array($result)) {
               array_push($arrData["data"], array(
                  "label" => $row["gname"],
                  "value" => $row["scores"]
                  )
               );
            }

            /*JSON Encode the data to retrieve the string containing the JSON representation of the data in the array. */

            $jsonEncodedData = json_encode($arrData);

    /*Create an object for the column chart using the FusionCharts PHP class constructor. Syntax for the constructor is ` FusionCharts("type of chart", "unique chart id", width of the chart, height of the chart, "div id to render the chart", "data format", "data source")`. Because we are using JSON data to render the chart, the data format will be `json`. The variable `$jsonEncodeData` holds all the JSON data for the chart, and will be passed as the value for the data source parameter of the constructor.*/

            $columnChart = new FusionCharts("column2D", "myFirstChart" , 600, 300, "chart-1", "json", $jsonEncodedData);

            // Render the chart
            $columnChart->render();

            // Close the database connection
            $dbhandle->close();
         }

      ?>

      <div id="chart-1"><!-- Fusion Charts will render here--></div>
    </center>

   </body>

</html>
<?php
require_once "dbconf.php";

    try {
        $pdo = new PDO("mysql:host=$host;dbname=$dbname", $username, $password);
        echo "Connected to $dbname at $host successfully.";
        $sql = "select * from adv1";
        $q = $pdo->query($sql);
        $q->setFetchMode(PDO::FETCH_ASSOC);
    } catch (PDOException $pe) {
        die("Could not connect to the database $dbname :" . $pe->getMessage());
    } ?>

    >
    <!DOCTYPE html>
    <html lang = "en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewpoint" content = "width = device-width, intial-scale = 1">
        <title> Sneaky Terror</title>
        <link rel="stylesheet" type="text/css" href="css/bootstrap.css">
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/boostrap.min.css">
        <link rel="stylesheet" href="http://getbootstrap.com/docs/4.1/examples/starter-template/starter-template.css">
        <link href='https://fonts.googleapis.com/css?family=Pacifico' rel='stylesheet' type='text/css'>
    <style>
        .jumbotron{
            background-color:#2E2D88;
            color:white
        }
        /*Adds borders for tabs*/
        .tab-content {
            border-left:1px solid #ddd;
            border-right:1px solid #ddd;
            border-bottom:1px solid #ddd;
            padding:10px;
        }
        .nav-tabs{
            margin-bottom:0;
        }
        table, th, td {
        border: 1px solid black;
        border-collapse: collapse;
        }
        th, td {
        padding: 5px;
        text-align: left;
        }
        .nav-tabs{
            margin-bottom:0;

        }
        li.dropdown {
        display: inline-block;
        }
        .dropdown-content {
            display: none;
            position: absolute;
            background-color: #f9f9f9;
            min-width: 160px;
            box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
            z-index: 1;
        }
        .dropdown-content a {
        color: black;
        padding: 12px 16px;
        text-decoration: none;
        display: block;
        text-align: left;
        }
        .dropdown-content a:hover {background-color: #f1f1f1}
        .dropdown:hover .dropdown-content {
        display: block;
        }
    </style>
    </head>
    <body>

      <nav class="navbar navbar-inverse navbar-fixed-top">
              <div class="container">
                <div class="navbar-header">
                  <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#navbar" aria-expanded="false" aria-controls="navbar">
                    <span class="sr-only">Toggle navigation</span>
                    <span class="icon-bar"></span>
                    <span class="icon-bar"></span>
                    <span class="icon-bar"></span>
                  </button>
                  <a class="navbar-brand" href="#">Sneaky Project</a>
                </div>
                <div id="navbar" class="navbar-collapse collapse" aria-expanded="false" style="height: 1px;">
                  <ul class="nav navbar-nav">
                    <<li><a href="#Home" onclick="location.href='./index.php';">Home</a></li>
                    <li class="dropdown">
                      <a href="javascript:void(0)" class="dropbtn">Basic Function</a>
                      <div class="dropdown-content">
                        <a href="#Display" onclick="location.href='./display.php';">Display victim place data</a>
                        <a href="#Insert" onclick="location.href='./display1.php';">Display details of terrorattack</a>
                        <a href="#Search" onclick="location.href='./search.php';">Search</a>
                      </div>
                    </li>
                    <li class="dropdown">
                        <a href="javascript:void(0)" class="dropbtn">Advanced Function</a>
                        <div class="dropdown-content">
                            <a href="#Advaned1" onclick="location.href='./adv1.php';">Gang Analysis and assessment</a>
                            	<a href="#Advaned2" onclick="location.href='./adv2.php';">How did Terrorism Spread (gif animation)</a>
                        </div>
                      </li>


                  </ul>

                </div><!--/.nav-collapse -->
              </div>
            </nav>

    <div class="container">


    <div class="page-header">
    <h1>Sneaky Terror</h1>
    </div>

    <div class = "jumbotron">

       <p>Top 10 most dangerous gangs based on the data so far</p>

       <p>
        </div>
        <table style="width:100%">
            <caption>Places</caption>
            <tr>
              <th>gname</th>
              <th>numissues</th>
                <th>nkill</th>
                  <th> attacktypes </th>
                    <th>scores</th>
              <th>action</th>
            </tr>
    <?php while ($row = $q->fetch()): ?>
            <tr>
              <td><?php echo htmlspecialchars($row['gname']); ?></td>
              <td><?php echo htmlspecialchars($row['numissues']); ?></td>
              <td><?php echo htmlspecialchars($row['nkill']); ?></td>
              <td><?php echo htmlspecialchars($row['attacktypes']); ?></td>
              <td><?php echo htmlspecialchars($row['scores']); ?></td>
                <td>
              <a href="detail.php?detail=<?php echo $row['gname']; ?>" class="detail_btn" >details</a>
            </td>

    			</td>


            <?php endwhile; ?>
          </tr>
          </table>
        </div>



       </p>


    </div>



    </div>

    <div class="row">

    </div>




    <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3"></script>
    <script src="js/bootstrap.min.js"></script>
    </body>
    </html>
