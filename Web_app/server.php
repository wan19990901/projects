<?php
	require_once "dbconf.php";
	session_start();
	$db = mysqli_connect($host, $username, $password, $dbname);

	// initialize variables
	$longtitude = "";
	$latitude = "";
	$eventid = "";
	$update = false;

	if (isset($_POST['save1'])) {
    $eventid = $_POST['eventid'];
		$latitude = $_POST['latitude'];
		$longtitude = $_POST['longtitude'];
			$year = $_POST['year'];
				$month = $_POST['month'];
					$day = $_POST['day'];
						$country = $_POST['country'];


		mysqli_query($db, "INSERT into places VALUES ('$eventid','$year','$month','$day','$country','$latitude','$longtitude')");

		header('location: display.php');
		$_SESSION['message1'] = "Address saved";
	}
	if (isset($_POST['save2'])) {
    $eventid = $_POST['eventid'];
		$summary = $_POST['summary'];
		$success = $_POST['success'];
			$attacktype = $_POST['attacktype'];
				$gname = $_POST['gname'];
					$nkill = $_POST['nkill'];
						$propvalue = $_POST['propvalue'];


		mysqli_query($db, "INSERT into terrorattack VALUES ('$eventid','$summary','$success','$attacktype','$gname','$nkill','$propvalue')");

		header('location: display1.php');
		$_SESSION['message3'] = "Address saved";
	}
  if (isset($_POST['update'])) {
  	$eventid = $_POST['eventid'];
  	$latitude = $_POST['latitude'];
  	$longtitude = $_POST['longtitude'];
		$year = $_POST['year'];
		$month = $_POST['month'];
		$day = $_POST['day'];
		$country = $_POST['country'];

  	mysqli_query($db, "UPDATE places SET longtitude='$longtitude', latitude='$latitude',year = '$year',month = '$month',day = '$day',country =
			'$country' WHERE eventid='$eventid'");
  	$_SESSION['message2'] = "Address updated!";
  	header('location: display.php');
  }
	if (isset($_POST['update1'])) {
		$eventid = $_POST['eventid'];
		$summary= $_POST['summary'];
		$success = $_POST['success'];
		$attacktype = $_POST['attacktype'];
		$gname = $_POST['gname'];
		$nkill = $_POST['nkill'];
		$propvalue = $_POST['propvalue'];

		mysqli_query($db, "UPDATE terrorattack SET summary='$summary', success='$success',attacktype = '$attacktype',gname = '$gname',propvalue = '$propvalue',nkill =
			'$nkill' WHERE eventid='$eventid'");
		$_SESSION['message2'] = "Address updated!";
		header('location: display1.php');
	}
	if (isset($_GET['del'])) {
	$eventid = $_GET['del'];
	mysqli_query($db, "DELETE FROM places WHERE eventid=$eventid");
	$_SESSION['message'] = "Address deleted!";
	header('location: display.php');
}

if (isset($_GET['del1'])) {
$eventid = $_GET['del1'];
mysqli_query($db, "DELETE FROM terrorattack WHERE eventid=$eventid");
$_SESSION['message'] = "Address deleted!";
header('location: display1.php');
}
