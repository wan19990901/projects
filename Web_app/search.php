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
<div class = "jumbotron">
   <h1>Sneaky Terror</h1>
        <p>Sneaky Terror is a database system that manages and displays information about global terrorist attacks. Any records in the database can be searched here.</p>
        <!--<p><a class="btn btn-primary btn-lg" href="#" role="button" onclick="location.href='./display.html';" >Check our data&raquo;</a></p>-->
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

<div class="row">

</div>

<?php
require_once 'dbconf.php';
$conn = mysqli_connect($host, $username, $password, $dbname);
$output = NULL;
$count = 0;

if(isset($_POST['placeSearch'])){

    $query = "SELECT * FROM places INNER JOIN terrorattack on places.eventid = terrorattack.eventid";
    $conditons = array();

    $eventid = ($_POST['event_id']);

    $year = ($_POST['year']);
    $month = ($_POST['month']);
    $day = ($_POST['day']);
    $country = ($_POST['country']);
    $latitude = ($_POST['latitude']);
    $longtitude = ($_POST['longtitude']);
    $summary = ($_POST['summary']);
    $success = ($_POST['success']);
    $attacktype = ($_POST['attacktype']);
    $gname = ($_POST['gname']);
    $nkill = ($_POST['nkill']);
    $propvalue = ($_POST['propvalue']);

    if ($eventid!= null){
        $conditions[] = "places.eventid = '$eventid'";
    }
    if ($year != null) {
        $conditions[] = "year = '$year'";
    }
    if ($month != null) {
        $conditions[] = "month = '$month'";
    }
    if ($day != null) {
        $conditions[] = "day = '$day'";
    }
    if ($country != null) {
        $conditions[] = " country = '$country'";
    }
    if ($latitude != null) {
        $conditions[] = "latitude = '$latitude'";
    }

    if ($longtitude != null) {
        $conditions[] = " longtitude = '$longtitude'";
    }

    if ($summary != null) {
        $conditions[] = " summary = '$summary'";
    }

    if ($success != null) {
        $conditions[] = " success = '$success'";
    }

    if ($attacktype != null) {
        $conditions[] = " attacktype = '$attacktype'";
    }

    if ($gname != null) {
        $conditions[] = " gname ='$gname'";
    }

    if ($nkill != null) {
        $conditions[] = " nkill = '$nkill'";
    }

    if ($propvalue != null) {
        $conditions[] = " propvalue = '$propvalue'";
    }

    $sql = $query;
    if (count($conditions) > 0) {
        $sql .= " where ".implode(' AND ',$conditions);


    }

    $result = mysqli_query($conn,$sql);


//Replace table_name with your table name and `thing_to_search` with the column you want to search
    $count = mysqli_num_rows($result);


}


?>


<script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3"></script>
<script src="js/bootstrap.min.js"></script>
</body>
</html>
<!DOCTYPE html>
<html>
	<head>
		<title>Search</title>
	</head>
	<body>
    <center>
    <form method="POST" action="search.php">

			<br> eventId &nbsp;<input type="text" name="event_id" placeholder=" "> </br>

            <br> year &nbsp;<input type="text" name="year"> </br>
            <br> month &nbsp;<input type="text" name="month" placeholder=" "> </br>
            <br> day &nbsp;<input type="text" name="day" placeholder=" "> </br>
            <br> country &nbsp;<input type="text" name="country" placeholder=" "> </br>
            <br> latitude &nbsp;<input type="text" name="latitude" placeholder=" "> </br>
            <br> longtitude &nbsp;<input type="text" name="longtitude" placeholder=" "> </br>

            <br> summary &nbsp;<input type="text" name="summary" placeholder=" "> </br>
            <br> success &nbsp;<input type="text" name="success" placeholder=" "> </br>
            <br> attacktype &nbsp;<input type="text" name="attacktype" placeholder=" "> </br>
            <br> gname &nbsp;<input type="text" name="gname" placeholder=" "> </br>
            <br> nkill &nbsp;<input type="text" name="nkill" placeholder=" "> </br>
            <br> propvalue &nbsp;<input type="text" name="propvalue" placeholder=" "> </br>


			<br> <input type="submit" name="placeSearch" value="Search"> </br>
		</form>

    <table width = "70%" cellpadding="5" cellspace="5">
        <tr>
            <td><strong>event_id</strong></td>

            <td><strong>year</strong></td>
            <td><strong>month</strong></td>
            <td><strong>day</strong></td>
            <td><strong>country</strong></td>
            <td><strong>latitude</strong></td>
            <td><strong>longtitude</strong></td>
            <td><strong>summary</strong></td>
            <td><strong>success</strong></td>
            <td><strong>attack_type</strong></td>
            <td><strong>g_name</strong></td>
            <td><strong>n_kill</strong></td>
            <td><strong>prop_value</strong></td>

        </tr>
        <?php
        if($count <= 0){
            $output = "NO RESULT";
        } else {
            while($rows = mysqli_fetch_array($result)) {
                    $eventid_row = $rows['eventid'];
                    $year_row = $rows['year'];
                    $month_row = $rows['month'];
                    $day_row = $rows['day'];
                    $country_row = $rows['country'];
                    $latitude_row = $rows['latitude'];
                    $longtitude_row = $rows['longtitude'];



                    $summary_row = $rows['summary'];
                    $success_row = $rows['success'];
                    $attacktype_row = $rows['attacktype'];
                    $gname_row = $rows['gname'];
                    $nkill_row = $rows['nkill'];
                    $propvalue_row = $rows['propvalue'];

            ?>
            <tr>
                <td><?php echo $eventid_row?></td>
                <td><?php echo $year_row?></td>
                <td><?php echo $month_row?></td>
                <td><?php echo $day_row?></td>
                <td><?php echo $country_row?></td>
                <td><?php echo $latitude_row?></td>
                <td><?php echo $longtitude_row?></td>
                <td><?php echo $summary_row?></td>
                <td><?php echo $success_row?></td>
                <td><?php echo $attacktype_row?></td>
                <td><?php echo $gname_row?></td>
                <td><?php echo $nkill_row?></td>
                <td><?php echo $propvalue_row?></td>
                </tr>



         <?php
	    }
}
?>

    </table>



  </center>


 <?php
 if($output == null) {
     echo $output;
 }

 ?>



	</body>
</html>
