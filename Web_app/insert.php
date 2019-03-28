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
                <li><a href="#Home" onclick="location.href='./index.html';">Home</a></li>
                <li class="dropdown">
                  <a href="javascript:void(0)" class="dropbtn">Basic Function</a>
                  <div class="dropdown-content">
                      <a href="#Display" onclick="location.href='./display.php';">Display</a>
                      <a href="#Display" onclick="location.href='./insert.php';">Insert</a>
                      <a href="#Delete" onclick="location.href='./delete.php';">Delete</a>
                      <a href="#Update" onclick="location.href='./update.php';">Update</a>
                      <a href="#Search" onclick="location.href='./search.php';">Search</a>
                  </div>
                </li>
                <li class="dropdown">
                    <a href="javascript:void(0)" class="dropbtn">Advanced Function</a>
                    <div class="dropdown-content">
                        <a href="#Display" onclick="location.href='./display.php';">Display</a>
                        <a href="#Insert" onclick="location.href='./insert.php';">Insert</a>
                        <a href="#Delete" onclick="location.href='./delete.php';">Delete</a>
                        <a href="#Update" onclick="location.href='./update.php';">Update</a>
                        <a href="#Search" onclick="location.href='./search.php';">Search</a>
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

   <p>#    I am a placeholder  :)    #</p>

   <p>
    </div>

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
/**
     * Insert three new places into the table
     Check display to see what is upcoming
     */
     <?php
require_once 'dbconf.php';

// Create connection
$conn = new mysqli($host, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

$sql = "INSERT INTO places (latitude,longtitude, Location,country)
VALUES (40,30, '','China');";
$sql .= "INSERT INTO places (latitude,longtitude, Location,country)
VALUES (20,70.4, '','Russia');";
$sql .= "INSERT INTO places (latitude,longtitude, Location,country)
VALUES (80.3,33.4, '','United States')";

if ($conn->multi_query($sql) === TRUE) {
    echo "New records created successfully";
} else {
    echo "Error: " . $sql . "<br>" . $conn->error;
}

$conn->close();
?>
