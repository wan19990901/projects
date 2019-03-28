<!--<html>
   <head>
      <title>Advanced Function 2</title>
   </head>
   <?php
      /*$output = system("python3.6 ./gif/make_gif.py");*/
   ?>
   <body>
      <img src="./gif/data/animation.gif" alt="The gif animation should be here.">
   </body>
</html>-->

</html>
<?php
/*require_once "dbconf.php";

    try {
        $pdo = new PDO("mysql:host=$host;dbname=$dbname", $username, $password);
        echo "Connected to $dbname at $host successfully.";
        $sql = "select * from adv1";
        $q = $pdo->query($sql);
        $q->setFetchMode(PDO::FETCH_ASSOC);
    } catch (PDOException $pe) {
        die("Could not connect to the database $dbname :" . $pe->getMessage());
    }*/ ?>

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
    <h1>How did Terrorism Spread?</h1>
    </div>

    <div class = "jumbotron">
       <p>The dataset containing terrorism information within the region and time you selected is visualized as follows. The first static figure shows the distribution of all the terrorist attacks in that dataset. The second animation shows the time evolution of the distribution of the terrorist attacks.</p>
    </div>
    
    <?php
       $var_reg=htmlentities($_POST['region'], ENT_QUOTES, "UTF-8");
       $var_starttime = $_POST['starttime'];
       $var_endtime = $_POST['endtime'];
       $command = "python3.6 ./gif/make_gif.py {$var_reg} {$var_starttime} {$var_endtime}";
       $output = system($command);
    ?>
       <center>
       <img src="./gif/data/staticfig.png" alt="The png figure should be here." width="1000", height="600"/>
       <img src="./gif/data/animation.gif" alt="The gif animation should be here."/>
       </center>

    <div class="row">

    </div>




    <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3"></script>
    <script src="js/bootstrap.min.js"></script>
    </body>
    </html>
