<!DOCTYPE html>
<?php

require_once 'dbconf.php';

try {
    $pdo = new PDO("mysql:host=$host;dbname=$dbname", $username, $password);
    /*echo "Connected to $dbname at $host successfully.";*/
    $sql = 'SELECT * FROM terrorattack Order By eventid DESC limit 50000';
    $q = $pdo->query($sql);
    $q->setFetchMode(PDO::FETCH_ASSOC);
} catch (PDOException $pe) {
    die("Could not connect to the database $dbname :" . $pe->getMessage());
} ?>

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

   <p>Please find details of terror attack data here, as well as editing or deleting a certain record.</p>

   <p>
    </div>


    <table style="width:100%">
        <caption>terrorattack</caption>
        <tr>
          <th> eventid </th>
          <th>summary</th>
            <th>success</th>
              <th>attacktype</th>
                <th>gname</th>
          <th>nkill</th>
          <th>propvalue</th>
          <th>action</th>

        </tr>
<?php while ($row = $q->fetch()): ?>
        <tr>
            <td><?php echo htmlspecialchars($row['eventid']); ?></td>
          <td><?php echo htmlspecialchars($row['summary']); ?></td>
          <td><?php echo htmlspecialchars($row['success']); ?></td>
          <td><?php echo htmlspecialchars($row['attacktype']); ?></td>
          <td><?php echo htmlspecialchars($row['gname']); ?></td>
          <td><?php echo htmlspecialchars($row['nkill']); ?></td>
          <td><?php echo htmlspecialchars($row['propvalue']); ?></td>

          <td>
				<a href="update1.php?edit1=<?php echo $row['eventid']; ?>" class="edit_btn" >Edit</a>
			</td>
			<td>
				<a href="server.php?del1=<?php echo $row['eventid']; ?>" class="del_btn">Delete</a>
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
