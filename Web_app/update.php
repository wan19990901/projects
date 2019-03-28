<?php  include('server.php'); ?>
<?php
	if (isset($_GET['edit'])) {
		$eventid = $_GET['edit'];
		$update = true;
		$record = mysqli_query($db, "SELECT * FROM places WHERE eventid=$eventid");
		if (count($record) == 1  ) {
			$n = mysqli_fetch_array($record);
			$latitude = $n['latitude'];
			$longtitude = $n['longtitude'];
			$year = $n['year'];
			$month = $n['month'];
				$day = $n['day'];
					$country = $n['country'];
		}
	}
?>
<!DOCTYPE html>
<html lang = "en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewpoint" content = "width = device-width, intial-scale = 1">
    <title> Sneaky Terror</title>
    <link rel="stylesheet" type="text/css" href="css/bootstrap.css">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/boostrap.min.css">
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
		.input-group{
			color: black;
	    padding: 12px 16px;
	    text-decoration: none;
	    display: block;
	    text-align:center;
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
                      <li><a href="#Home" onclick="location.href='./index.php';">Home</a></li>
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
<div class = "jumbotron">
   <h1>Under Construction</h1>
        <p>#    I am a placeholder  :)    #</p>
        <p><a class="btn btn-primary btn-lg" href="#" role="button" onclick="location.href='./display.html';" >Check our data&raquo;</a></p>
      </div>
    </div>
    <div class="container">
      <!-- Example row of columns -->
      <div class="row">
        <div class="col-md-4">
          <h2>Basic Function</h2>
          <p> Includes insert,update,search and display function. </p>
          <p><a class="btn btn-default" href="#" role="button">View details &raquo;</a></p>

        </div>
        <div class="col-md-4">
          <h2>Advaned Function</h2>
          <p>Under construction </p>
          <p><a class="btn btn-default" href="#" role="button">View details &raquo;</a></p>
       </div>
        <div class="col-md-4">
          <h2>Placeholder</h2>
          <p>Under construction</p>
          <p><a class="btn btn-default" href="#" role="button">View details &raquo;</a></p>
        </div>
      </div>
      <hr>
      <footer>
        <p>&copy; 2018 Team, 4AM.</p>
      </footer>
</div>
</div>

<div class="row">
</div>

<script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3"></script>
<script src="js/bootstrap.min.js"></script>
</body>
</html>
<html>
<head>
	<title>CRUD: CReate, Update, Delete PHP MySQL</title>
</head>
<body>
  <body>

	<form method="post" action="server.php" >
		<div class="input-group">
			<label>eventid</label>
			<input type="hidden" name="eventid" value="<?php echo $eventid; ?>">
		</div>
		<div class="input-group">
			<label>year</label>
			<input type="text" name="year" value="<?php echo $year; ?>">
		</div>
		<div class="input-group">
			<label>month</label>
			<input type="text" name="month" value="<?php echo $month; ?>">
		</div>
		<div class="input-group">
			<label>day</label>
			<input type="text" name="day" value="<?php echo  $day; ?>">
		</div>
		<div class="input-group">
			<label>country</label>
			<input type="text" name="country" value="<?php echo $country; ?>">
		</div>
		<div class="input-group">
			<label>latitude</label>
			<input type="text" name="latitude" value="<?php echo $latitude; ?>">
		</div>
    <div class="input-group">
      <label>longtitude</label>
      <input type="text" name="longtitude" value="<?php echo $longtitude; ?>">
    </div>
		<div class="input-group">

	<button class="btn" type="submit" name="update" style="background: #556B2F;" >update</button>
		</div>
	</form>
</body>
</html>
