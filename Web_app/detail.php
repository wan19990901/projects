<?php  include('server.php'); ?>
<?php
	if (isset($_GET['detail'])) {
       $record = mysqli_query($db, "SELECT * FROM terrorattack t WHERE t.gname = '{$_GET['detail']}'");
       if (!$record) {
     printf("Error: %s\n", mysqli_error($db));
     exit();

	}
	$record1 = mysqli_query($db, "SELECT sum(propvalue) as sum from terrorattack WHERE gname = '{$_GET['detail']}'
	ORDER BY sum DESC
	LIMIT 1");
	$record2 = mysqli_query($db, "SELECT count(success) as sum from terrorattack WHERE gname = 'Taliban' GROUP By success Having success > 0");
	if (!$record2) {
	printf("Error: %s\n", mysqli_error($db));
	exit();
	}

$record4 = mysqli_query($db, "SELECT attacktype, count(*) as sum from terrorattack WHERE gname = '{$_GET['detail']}'  GROUP BY attacktype
ORDER BY sum DESC
LIMIT 1");
if (!$record4) {
printf("Error: %s\n", mysqli_error($db));
exit();
}
$record5 = mysqli_query($db, "SELECT eventid,nkill from terrorattack WHERE gname = '{$_GET['detail']}'
ORDER BY nkill DESC
LIMIT 1");

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

table {
  table-layout: fixed;
  width: 100%;
  border-collapse: collapse;
  border: 3px solid purple;
}

thead th:nth-child(1) {
  width: 30%;
}

thead th:nth-child(2) {
  width: 20%;
}

thead th:nth-child(3) {
  width: 15%;
}

thead th:nth-child(4) {
  width: 35%;
}

th, td {
  padding: 20px;
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
				<h2>  Quick facts of Gang </h2>
<?php

while ($row = mysqli_fetch_array($record4)){
	echo 'the most frequent attack_type they use and its total count:'.'<tr>
	<td>'.$row['attacktype'].'</td>
	<td>'.$row['sum'].'</td>
	</tr><br>';
}

while ($row = mysqli_fetch_array($record1)){
	echo 'the total property value lost due to the Terrorism:'.'<tr>
	<td>'.'$'.$row['sum'].'</td>
	</tr><br>';
}
while ($row = mysqli_fetch_array($record2)){
	echo 'the total successful terrorattacks by the Terrorism:'.'<tr>
	<td>'.$row['sum'].'</td>
	</tr><br>';
}



while ($row = mysqli_fetch_array($record5)){
	echo 'the eventid for the most disastrous attack and its number of kills:'.'<tr>
	<td>'.$row['eventid'].'</td>
	<td>'.$row['nkill'].'</td>
	</tr><br>';
} ?>


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
    <table style="width:100%" class = 't'>
        <caption>terrorattack</caption>
        <tr>
          <th> gname     </th>
          <th>eventid</th>
					<th>summary</th>
						<th>success</th>
							<th>attacktype</th>
					<th>nkill</th>
					<th>propvalue</th>
        </tr>

        		<tbody>


<?php
        		while ($row = mysqli_fetch_array($record))
        		{

        			echo '<tr>
        					<td>'.$row['gname'].'</td>
        					<td>'.$row['eventid'].'</td>
        					<td>'.$row['summary'].'</td>
        					<td>'. $row['success'] . '</td>
        					<td>'.$row['attacktype'].'</td>
                  	<td>'.$row['nkill'].'</td>
                    	<td>'.$row['propvalue'].'</td>
        				</tr>';
              }
?>


        		</tbody>

        	</table>
        </body>
</body>
</html>
