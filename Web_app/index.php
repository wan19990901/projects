<?php  include('server.php'); ?>
<?php
	if (isset($_GET['edit'])) {
		$eventid = $_GET['edit'];
		$update = true;
		$record = mysqli_query($db, "SELECT * FROM places WHERE eventid=$eventid");

		if (count($record) == 1 ) {
			$n = mysqli_fetch_array($record);
			$latitude = $n['latitude'];
			$longtitude = $n['longtitude'];
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
                      <li class="active"><a href="#">Home</a></li>
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
   <h1>Sneaky Terror</h1>
        <p>Sneaky Terror is a database system that manages and displays information about global terrorist attacks. New records can be inserted in this page.</p>
        <!--<p><a class="btn btn-primary btn-lg" href="#" role="button" onclick="location.href='./detail.php';" >Check our data&raquo;</a></p>-->
      </div>
    </div>
    <div class="container">
      <!-- Example row of columns -->
      <hr>
      <footer>
        <p>&copy; 2018 Team, 4AM.</p>
      </footer>
</div>
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
  <body >
		<h3> Place to insert new records to places </h3>

	<form method="post" action="server.php" >
		<center>
		<div class="input-group">
			<label>eventid</label>
			<input type="text" name="eventid" value="">
		</div>
		<div class="input-group">
			<label>latitude</label>
			<input type="text" name="latitude" value="">
		</div>
    <div class="input-group">
      <label>longtitude</label>
      <input type="text" name="longtitude" value="">
    </div>
		<div class="input-group">
			<label>year</label>
			<input type="text" name="year" value="">
		</div>
		<div class="input-group">
			<label>month</label>
			<input type="text" name="month" value="">
		</div>
		<div class="input-group">
			<label>day</label>
			<input type="text" name="day" value="">
		</div>
		<div class="input-group">
			<label>country</label>
			<input type="text" name="country" value="">
		</div>
		<div class="input-group">

	<button class="btn" type="submit" name="save1" >Save</button>

		</div>
	</center>
	</form>
		<h3> Place to insert new records to terrorattack </h3>
	<form method="post" action="server.php" >
		<center>
		<div class="input-group">
			<label>eventid</label>
			<input type="text" name="eventid" value="">
		</div>
		<div class="input-group">
			<label>summary</label>
			<input type="text" name="summary" value="">
		</div>
    <div class="input-group">
      <label>success</label>
      <input type="text" name="success" value="">
    </div>
		<div class="input-group">
			<label>attacktype</label>
			<input type="text" name="attacktype" value="">
		</div>
		<div class="input-group">
			<label>gname</label>
			<input type="text" name="gname" value="">
		</div>
		<div class="input-group">
			<label>nkill</label>
			<input type="text" name="nkill" value="">
		</div>
		<div class="input-group">
			<label>propvalue</label>
			<input type="text" name="propvalue" value="">
		</div>
		<div class="input-group">

	<button class="btn" type="submit" name="save2" >Save</button>

		</div>
	</center>
	</form>
</body>
</html>
