function change_game() {
    new_game = document.getElementById("game").value;
    $('#set option').remove();
    get_interval_set(new_game);
    change_set();
}

function change_set(){
    game = document.getElementById("game").value;
    new_set = document.getElementById("set").value;
    $('#rally option').remove();
    change_rally();
}

function change_rally(){
    var game = document.getElementById("game").value;
    var set = document.getElementById("set").value;
    if(!set){
        set = 1;
    }
    //delete old canvas
    $('#canvas').remove();
    $('#balltype_table').remove();
    $('.btn').remove();

    get_interval_rally(set,game);
    init_trajectory(set,game);
}

function get_interval_game(){
    filename = 'statistics/game_name.json';
    $.getJSON(filename, function(data) {
        for(var i=1;i<=data.length;i++){
            var insertText = '<option value=' + data[i-1] + '>' + 'Game ' + i + '</option>';
            $('#game').append(insertText); 
        }
    })
}

function get_interval_set(game_name){
    //init game
    if (!game_name){
        game_name = "18IND_TC";
    }

    filename = 'statistics/rally_detail_real_' + game_name + '.json';

    $.getJSON(filename, function(data) {
        //find maximum set
        maximum = Math.max.apply(Math, data.map(function(d) { 
            return d.set; 
        }));

        for(var i=1;i<=maximum;i+=1)
        {
            var insertText = '<option value='+i+'>'+i+'</option>';
            $('#set').append(insertText); 
        }
    });
}

function get_interval_rally(set,game_name){
    var insertText = '<button id="interval-submit" type="button" class="btn btn-primary" onclick=change_rally()>查詢 <i class="fab fa-sistrix"></i></button>';
    $('#dropdown').append(insertText); 
    var insertText = '<button class="btn btn-default" id="next" type="button">下一球</button>';
    $('#dropdown').append(insertText); 
    var insertText = '<button class="btn btn-default" id="back" type="button">上一球</button>';
    $('#dropdown').append(insertText); 

    //init game
    if (!game_name){
        game_name = "18IND_TC";
    }
    filename = 'statistics/rally_detail_real_' + game_name + '.json';

    $.getJSON(filename, function(data) {
        //init set
        if (!set){
            set = 1;
        }
        //filter data to specific set
        data = data.filter(function(item) {
            return item.set == set
        });
        data = data[0].info;
        var maximum;
        maximum = Math.max.apply(Math, data.map(function(d) { 
            return d.rally; 
        }));
        
        for(var i=0;i<maximum;i+=1)
        {
            var score = data[i].score;
            var insertText = '<option value=' + (i+1) + '>' + score + ' (' + (i+1) + ')' + '</option>';
            $('#rally').append(insertText); 
        } 
    })
}

function init_trajectory(set,game_name){
    var cwidth = "1200";
    var cheight = "600";
    $('.ball_trajectory').html('<canvas id="canvas" width=' + cwidth + ' height=' + cheight + '></canvas>');
    var rally = document.getElementById("rally").value;
    var canvas = document.getElementById('canvas');
    var ctx = canvas.getContext('2d');
    var current = 0;
    var currentTableIdx = 0;
    var total_y_length = 424;
    var TopLeftX = 100;
    var TopLeftY = 100;
    var CourtW = 935;
    var CourtH = 424;
    ctx.clearRect(TopLeftX,TopLeftY,CourtW,CourtH);
    
    //init game
    if (!game_name){
        game_name = "18IND_TC";
    }
    filename = 'statistics/rally_detail_real_' + game_name + '.json';

    $.getJSON(filename, function(data) {
        if(!set){
            set = 1;
        }
        if(!rally){
            rally = 1;
        }

        //filter data to specific set
        data = data.filter(function(item) {
            return item.set == set
        });
        data = data[0].info;
        //filter data to specific rally
        data = data.filter(function(item) {
            return parseInt(item.rally) == rally;
        });
        data = data[0].result;
        console.log(data);
        var maxorder = Math.max.apply(Math, data.map(function(d) { 
            return d.order; 
        }));

        //balltype table initial
        $('#balltype_table').remove();
        var insertText = '<table class="table table-bordered" id="balltype_table"><thead><tr><th>Balltpye</th></tr></thead><tbody class="tbody_detail"></tbody></table>';
        $('.ball_trajectory').append(insertText); 

        function initial() {
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.strokeStyle = "black";
            //球場外框
            ctx.rect(TopLeftX,TopLeftY,CourtW,CourtH);
            //直的線
            ctx.moveTo(TopLeftX+53,TopLeftY);
            ctx.lineTo(TopLeftX+53,TopLeftY+CourtH);
            ctx.moveTo(TopLeftX+190,TopLeftY);
            ctx.lineTo(TopLeftX+190,TopLeftY+CourtH);
            ctx.moveTo(TopLeftX+328,TopLeftY);
            ctx.lineTo(TopLeftX+328,TopLeftY+CourtH);
            ctx.moveTo(TopLeftX+468,TopLeftY);
            ctx.lineTo(TopLeftX+468,TopLeftY+CourtH);
            ctx.moveTo(TopLeftX+608,TopLeftY);
            ctx.lineTo(TopLeftX+608,TopLeftY+CourtH);
            ctx.moveTo(TopLeftX+745,TopLeftY);
            ctx.lineTo(TopLeftX+745,TopLeftY+CourtH);	
            ctx.moveTo(TopLeftX+882,TopLeftY);
            ctx.lineTo(TopLeftX+882,TopLeftY+CourtH);
            //橫的線
            ctx.moveTo(TopLeftX,TopLeftY+32);
            ctx.lineTo(TopLeftX+CourtW,TopLeftY+32);
            ctx.moveTo(TopLeftX,TopLeftY+392);
            ctx.lineTo(TopLeftX+CourtW,TopLeftY+392);
            ctx.moveTo(TopLeftX,TopLeftY+212);
            ctx.lineTo(TopLeftX+328,TopLeftY+212);
            ctx.moveTo(TopLeftX+608,TopLeftY+212);
            ctx.lineTo(TopLeftX+CourtW,TopLeftY+212);
            ctx.closePath();
            ctx.stroke();
        }

        function CheckSmash(point){
            if(point.detail_type == '殺球'){
                return true;
            }
            return false;
        }
        
        initial();
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(data[0].detail_hit_pos[1]+100,total_y_length-data[0].detail_hit_pos[0]+100,5,0,Math.PI*2,true);
        ctx.strokeStyle = "black";
        ctx.closePath();
        ctx.stroke();
        $("#next").click(function(){    
            //add next balltype to table
            if(currentTableIdx != maxorder-1){
                var insertText;
                if(CheckSmash(data[currentTableIdx])){
                    insertText = '<tr class="success"><td><b>' + data[currentTableIdx].detail_type + '</b></td></tr>';
                }
                else{
                    insertText = '<tr><td>' + data[currentTableIdx].detail_type + '</td></tr>';
                }
                $('.tbody_detail').append(insertText); 
                currentTableIdx += 1;
            }

            if(current != maxorder-1){
                if(current>2) {
                    ctx.beginPath();
                    ctx.clearRect(50,50,1000,600);
                    ctx.closePath();
                    ctx.stroke();
                    initial();
                    //faded
                    ctx.lineWidth = 3;
                    for(var j=0;j<current-2;j++) {
                        ctx.beginPath();
                        ctx.arc(data[j].detail_hit_pos[1]+100,total_y_length-data[j].detail_hit_pos[0]+100,5,0,Math.PI*2,true);
                        ctx.strokeStyle = "rgb(229, 226, 222)";
                        ctx.closePath();
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.moveTo(data[j].detail_hit_pos[1]+100,total_y_length-data[j].detail_hit_pos[0]+100);
                        ctx.lineTo(data[j+1].detail_hit_pos[1]+100,total_y_length-data[j+1].detail_hit_pos[0]+100);
                        ctx.strokeStyle = "rgb(229, 226, 222)";
                        ctx.closePath();
                        ctx.stroke();
                    }
                    //normal
                    for(var j=current-2;j<current+1;j++) {
                        ctx.beginPath();
                        ctx.arc(data[j].detail_hit_pos[1]+100,total_y_length-data[j].detail_hit_pos[0]+100,5,0,Math.PI*2,true);
                        ctx.strokeStyle = "black";
                        ctx.closePath();
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.moveTo(data[j].detail_hit_pos[1]+100,total_y_length-data[j].detail_hit_pos[0]+100);
                        ctx.lineTo(data[j+1].detail_hit_pos[1]+100,total_y_length-data[j+1].detail_hit_pos[0]+100);
                        if(j==current-2){
                            if(CheckSmash(data[j])){
                                ctx.strokeStyle = "rgb(66, 245, 147)";
                            }
                            else{
                                ctx.strokeStyle = "rgb(252, 133, 133)";
                            }
                        }
                        if(j==current-1){
                            if(CheckSmash(data[j])){
                                ctx.strokeStyle = "rgb(66, 245, 147)";
                            }
                            else{
                                ctx.strokeStyle = "rgb(173, 34, 34)";
                            }
                        }
                        if(j==current){
                            if(CheckSmash(data[j])){
                                ctx.strokeStyle = "rgb(66, 245, 147)";
                            }
                            else{
                                ctx.strokeStyle = "rgb(91, 0, 0)";
                            }
                        }
                        ctx.closePath();
                        ctx.stroke();
                    }
                    ctx.beginPath();
                    ctx.arc(data[current+1].detail_hit_pos[1]+100,total_y_length-data[current+1].detail_hit_pos[0]+100,5,0,Math.PI*2,true);
                    ctx.strokeStyle = "black";
                    ctx.closePath();
                    ctx.stroke();
                }
                else {
                    ctx.beginPath();
                    ctx.clearRect(50,50,1000,600);
                    ctx.closePath();
                    ctx.stroke();
                    initial();
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.arc(data[0].detail_hit_pos[1]+100,total_y_length-data[0].detail_hit_pos[0]+100,5,0,Math.PI*2,true);
                    ctx.strokeStyle = "black";
                    ctx.closePath();
                    ctx.stroke();
                    for(var j=current+1;j>0;j--) {
                        ctx.beginPath();
                        ctx.arc(data[j].detail_hit_pos[1]+100,total_y_length-data[j].detail_hit_pos[0]+100,5,0,Math.PI*2,true);
                        ctx.strokeStyle = "black";
                        ctx.closePath();
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.moveTo(data[j].detail_hit_pos[1]+100,total_y_length-data[j].detail_hit_pos[0]+100);
                        ctx.lineTo(data[j-1].detail_hit_pos[1]+100,total_y_length-data[j-1].detail_hit_pos[0]+100);
                        if(j==current-1){
                            if(CheckSmash(data[j-1])){
                                ctx.strokeStyle = "rgb(66, 245, 147)";
                            }
                            else{
                                ctx.strokeStyle = "rgb(252, 133, 133)";
                            }
                        }
                        if(j==current){
                            if(CheckSmash(data[j-1])){
                                ctx.strokeStyle = "rgb(66, 245, 147)";
                            }
                            else{
                                ctx.strokeStyle = "rgb(173, 34, 34)";
                            }
                        }
                        if(j==current+1){
                            if(CheckSmash(data[j-1])){
                                ctx.strokeStyle = "rgb(66, 245, 147)";
                            }
                            else{
                                ctx.strokeStyle = "rgb(91, 0, 0)";
                            }
                        }
                        ctx.closePath();
                        ctx.stroke();
                    }
                }
                if(current!=maxorder-2) {
                    current+=1;
                }
                else{
                    current = maxorder - 1;
                }
            }
            
        });
    
        $("#back").click(function(){
            //delete last table row
            var table = document.getElementById('balltype_table');
            var rowCount = table.rows.length;
            if(currentTableIdx > 0){
                table.deleteRow(-1);
                currentTableIdx -= 1;
            }

            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.clearRect(50,50,1000,600);
            ctx.closePath();
            ctx.stroke();
            initial();
            ctx.lineWidth = 3;
            if(current>4) {
                //faded
                for(var j=0;j<current-4;j++) {
                    ctx.beginPath();
                    ctx.arc(data[j].detail_hit_pos[1]+100,total_y_length-data[j].detail_hit_pos[0]+100,5,0,Math.PI*2,true);
                    ctx.strokeStyle = "rgb(229, 226, 222)";
                    ctx.closePath();
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(data[j].detail_hit_pos[1]+100,total_y_length-data[j].detail_hit_pos[0]+100);
                    ctx.lineTo(data[j+1].detail_hit_pos[1]+100,total_y_length-data[j+1].detail_hit_pos[0]+100);
                    ctx.strokeStyle = "rgb(229, 226, 222)";
                    ctx.closePath();
                    ctx.stroke();
                }
                //normal
                for(var j=current-4;j<current-1;j++) {
                    ctx.beginPath();
                    ctx.arc(data[j].detail_hit_pos[1]+100,total_y_length-data[j].detail_hit_pos[0]+100,5,0,Math.PI*2,true);
                    ctx.strokeStyle = "black";
                    ctx.closePath();
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(data[j+1].detail_hit_pos[1]+100,total_y_length-data[j+1].detail_hit_pos[0]+100);
                    ctx.lineTo(data[j].detail_hit_pos[1]+100,total_y_length-data[j].detail_hit_pos[0]+100);
                    if(j==current-4){
                        if(CheckSmash(data[j])){
                            ctx.strokeStyle = "rgb(66, 245, 147)";
                        }
                        else{
                            ctx.strokeStyle = "rgb(252, 133, 133)";
                        }
                    }
                    if(j==current-3){
                        if(CheckSmash(data[j])){
                            ctx.strokeStyle = "rgb(66, 245, 147)";
                        }
                        else{
                            ctx.strokeStyle = "rgb(173, 34, 34)";
                        }
                    }
                    if(j==current-2){
                        if(CheckSmash(data[j])){
                            ctx.strokeStyle = "rgb(66, 245, 147)";
                        }
                        else{
                            ctx.strokeStyle = "rgb(91, 0, 0)";
                        }
                    }
                    
                    ctx.closePath();
                    ctx.stroke();
                }
                ctx.beginPath();
                ctx.arc(data[current-1].detail_hit_pos[1]+100,total_y_length-data[current-1].detail_hit_pos[0]+100,5,0,Math.PI*2,true);
                ctx.strokeStyle = "black";
                ctx.closePath();
                ctx.stroke();
            }
            else {
                ctx.beginPath();
                ctx.arc(data[0].detail_hit_pos[1]+100,total_y_length-data[0].detail_hit_pos[0]+100,5,0,Math.PI*2,true);
                ctx.strokeStyle = "black";
                ctx.closePath();
                ctx.stroke();

                for(var j=current-1;j>=1;j--) {
                    ctx.beginPath();
                    ctx.arc(data[j].detail_hit_pos[1]+100,total_y_length-data[j].detail_hit_pos[0]+100,5,0,Math.PI*2,true);
                    ctx.strokeStyle = "black";
                    ctx.closePath();
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(data[j].detail_hit_pos[1]+100,total_y_length-data[j].detail_hit_pos[0]+100);
                    ctx.lineTo(data[j-1].detail_hit_pos[1]+100,total_y_length-data[j-1].detail_hit_pos[0]+100);
                    if(j==current-3){
                        if(CheckSmash(data[j-1])){
                            ctx.strokeStyle = "rgb(66, 245, 147)";
                        }
                        else{
                            ctx.strokeStyle = "rgb(252, 133, 133)";
                        }
                    }
                    if(j==current-2){
                        if(CheckSmash(data[j-1])){
                            ctx.strokeStyle = "rgb(66, 245, 147)";
                        }
                        else{
                            ctx.strokeStyle = "rgb(173, 34, 34)";
                        }
                    }
                    if(j==current-1){
                        if(CheckSmash(data[j-1])){
                            ctx.strokeStyle = "rgb(66, 245, 147)";
                        }
                        else{
                            ctx.strokeStyle = "rgb(91, 0, 0)";
                        }
                    }
                    ctx.closePath();
                    ctx.stroke();
                }
            }
            if(current!=0) {
                current-=1;
            }
        })
    });
}