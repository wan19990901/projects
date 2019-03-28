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
       <p>Welcome! This function provides a straightforward way to see how the terrorism spreaded in a certain region within a specific time inteval by projecting the attacks into real-world maps. Please specify the region and time inteval you are interested in.</p>
    </div>

    <p>
    <center>
    <form action="adv2_action.php" method="post">
    Select Region
       <select name="region">
       <option value="World">World</option>
       <option value="Afghanistan">Afghanistan</option>
       <option value="Argentina">Argentina</option>
       <option value="Australia">Australia</option>
       <option value=<"Bolivia">Bollivia</option>
       <option value="Brazil">Brazil</option>
       <option value="Canada">Canada</option>
       <option value="China">China</option>
       <option value="Colombia">Colombia</option>
       <option value="Egypt">Egypt</option>
       <option value="Ethiopia">Ethiopia</option>
       <option value="France">France</option>
       <option value="Germany">Germany</option>
       <option value="Greece">Greece</option>
       <option value="India">India</option>
       <option value="Indonesia">Indonesia</option>
       <option value="Iran">Iran</option>
       <option value="Iraq">Iraq</option>
       <option value="World">Israel</option>
       <option value="Italy">Italy</option>
       <option value="Japan">Japan</option>
       <option value="Lebanon">Lebanon</option>
       <option value="Malaysia">Malaysia</option>
       <option value="Mexico">Mexico</option>
       <option value="North_Korea">North Korea</option>
       <option value="New_Zealand">New Zealand</option>
       <option value="Parkistan">Parkistan</option>
       <option value="Philippines">Philippines</option>
       <option value="Russia">Russia</option>
       <option value="Saudi Arabia">Saudi Arabia</option>
       <option value="Somalia">Somalia</option>
       <option value="South_Africa">South Africa</option>
       <option value="South_Korea">South Korea</option>
       <option value="Soviet_Union">Soviet Union</option>
       <option value="Spain">Spain</option>
       <option value="Syria">Syria</option>
       <option value="Turkey">Turkey</option>
       <option value="United_Kingdom">United Kingdom</option>
       <option value="United_States">United States</option>
       <option value="Uruguay">Uruguay</option>
       <option value="Vietnam">Vietnam</option>
       <option value="Zambia">Zambia</option>
       </select><br>
    Start Year (after 1970-01-01)
       <input type="date" name="starttime" min="1970-01-01"><br>
    End Year (before 2016-12-31)
       <input type="date" name="endtime" max="2016-12-31"><br>
       <input type="submit" value="Visualize" name="Submit">
    </form>
    </center>
    </p>

    <div class="row">

    </div>




    <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3"></script>
    <script src="js/bootstrap.min.js"></script>
    </body>
    </html>
