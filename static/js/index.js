$(document).ready(function(){
    //connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port + '/test');
    var numbers_received = [];

    //receive details from server
    socket.on('newdata', function(msg) {
        console.log("Received " + msg.data);
        $('#data').html(msg.data);
    });

});