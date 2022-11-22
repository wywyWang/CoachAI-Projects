
function init_linechart(minrally,maxrally,set,game_name){
    //init game
    if (!game_name){
        game_name = "18IND_TC";
    }
    filename = 'statistics/rally_count_real_' + game_name + '.json';

    $.getJSON(filename, function(data) {
        [data,set,minrally,maxrally] = data_filter(data,set,minrally,maxrally,0);

        //init svg legend
        d3.selectAll("svg").remove();
        var svg_legend = d3.select("#line").append("svg")
                            .attr("width", '100%')
                            .attr("height", '5vh')

        var posx = 150;
        var posy = 20;
        svg_legend.append("circle").attr("cx",posx).attr("cy",posy).attr("r", 6).style("fill", "rgb(66,129,164)")
        svg_legend.append("circle").attr("cx",posx+200).attr("cy",posy).attr("r", 6).style("fill", "rgb(255,99,132)")
        svg_legend.append("text").attr("class", "d3_legend").attr("x", posx+10).attr("y", posy)
                    .text("Player A Win").style("fill","rgb(66,129,164)").attr("alignment-baseline","middle")
        svg_legend.append("text").attr("class", "d3_legend").attr("x", posx+200+10).attr("y", posy)
                    .text("Player B Win").style("fill","rgb(255,99,132)").attr("alignment-baseline","middle")

        var canv = document.createElement('canvas');
        canv.id = 'line_chart';
        canv.width = 640;
        canv.height = 360;
        document.getElementById("line").appendChild(canv);

        var chartRadarDOM;
        var chartRadarOptions;

        chartRadarDOM = document.getElementById("line_chart");
        //custormized options
        chartRadarOptions = 
        {
            legend:{
                display: false
            },
            scales:{
                xAxes: [{
                    scaleLabel:{
                        display: true,
                        labelString: '回合',
                        fontSize: 16
                    }
                }],
                yAxes: [{
                    ticks:{
                        beginAtZero: true,
                    },
                    scaleLabel:{
                        display: true,
                        labelString: '拍數',
                        fontSize: 16
                    }
                }]
            },
            elements: {
                line: {
                    tension: 0 // disables bezier curves
                }
            },
            animation: {
              duration: 1,
              onComplete: function() {
                var chartInstance = this.chart,
                ctx = chartInstance.ctx;

                ctx.font = Chart.helpers.fontString(Chart.defaults.global.defaultFontSize, Chart.defaults.global.defaultFontStyle, Chart.defaults.global.defaultFontFamily);
                ctx.textAlign = 'center';
                ctx.textBaseline = 'bottom';
                ctx.fillStyle = 'rgba(0,0,0,1)';

                this.data.datasets.forEach(function(dataset, i) {
                  var meta = chartInstance.controller.getDatasetMeta(i);
                  meta.data.forEach(function(bar, index) {
                    var data = dataset.data[index];
                    ctx.fillText(data, bar._model.x, bar._model.y - 5);
                  });
                });
              }
            }
        };

        var labels = data.map(function(e) {
            return e.rally;
        });

        var datas = data.map(function(e) {
            return e.stroke;
        });

        var pointcolor = [];
        var datadown = [];
        var datainterval = [];
        var dataup = [];
        for (var i = 0;i<data.length;i++){
            if (data[i].rally < minrally){
                datadown.push(data[i].stroke);
                datainterval.push(null);
                dataup.push(null);
            }
            else if (data[i].rally > maxrally){
                datadown.push(null);
                datainterval.push(null);
                dataup.push(data[i].stroke);
            }
            else{
                if (data[i].rally == minrally){
                    datadown.push(data[i].stroke);
                    datainterval.push(data[i].stroke);
                    dataup.push(null);
                }
                else if (data[i].rally == maxrally){
                    datadown.push(null);
                    datainterval.push(data[i].stroke);
                    dataup.push(data[i].stroke);
                }
                else{
                    datadown.push(null);
                    datainterval.push(data[i].stroke);
                    dataup.push(null);
                }
            }
                
            if (data[i].rally < minrally || data[i].rally > maxrally)
                pointcolor.push("rgb(216, 212, 212)");
            else if(data[i].winner == 'A')
                pointcolor.push("rgb(66,129,164)");
            else
                pointcolor.push("rgb(255,99,132)");
        }

        var segment_data=[];
        var consec_point=[];
        for (var i = 0;i<data.length-1;) {
            var point_data=[];
            if (data[i].rally < minrally || data[i].rally >= maxrally) {
                i+=1;
                continue;
            }
            if(i==data.length-2){
                point_data.push(i);
                point_data.push(i+1);
                segment_data.push(point_data);
                break;
            }
            
            if(data[i].winner==data[i+1].winner){
                consec_point.push(i);
                while(data[i].winner==data[i+1].winner){
                    if(i>=data.length-2){
                        point_data.push(i);
                        i+=1;
                        break;
                    }
                    point_data.push(i);
                    i+=1;
                }
                point_data.push(i);
            }
            else if(data[i].winner!=data[i+1].winner){
                while(data[i].winner!=data[i+1].winner){
                    if(i>=data.length-2){
                        point_data.push(i);
                        i+=1;
                        break;
                    }
                    point_data.push(i);
                    i+=1;
                }
                point_data.push(i);
            }
            segment_data.push(point_data);
            if(i==data.length-1){
                break;
            }
        }
        var final_data=new Array();
        for (var i = 0;i<data.length;i++) {
            final_data[i]=new Array();
        }
        for (var i = 0;i<segment_data.length;i++) {
            for(var j = 0;j<data.length;j++){
                var match=0;
                for(var k=0;k<segment_data[i].length;k++){
                    if(j==segment_data[i][k]){
                        match=1;
                        break;
                    }
                }
                if(match==1){
                    final_data[i].push(data[j].stroke);
                }
                else{
                    final_data[i].push(null);
                }                   
            }
        }
        var present_data=[];
        for (var i = 0;i<segment_data.length;i++) {
            var linecolor;
            for(var j=0;j<consec_point.length;j++){
                if(segment_data[i][0]==consec_point[j]){
                	if(data[segment_data[i][0]].winner=='A'){
                		linecolor="rgba(66,129,164,"+segment_data[i].length*0.2+")";
                	}               			
                	if(data[segment_data[i][0]].winner=='B')    {
                		linecolor="rgba(255,99,132,"+segment_data[i].length*0.2+")";
                	}               		
                    //"rgb("+(255-segment_data[i].length*10)+","+(236-segment_data[i].length*10)+","+(203-segment_data[i].length*10)+")";
                    break;
                }
                else{
                    linecolor="rgb(255, 236, 203)";
                }
            }
            
            present_data.push({fill: false,
                        cubicInterpolationMode:"monotone",
                        backgroundColor: "rgba(66,129,164,0.2)",
                        borderColor: linecolor,
                        pointBorderColor: "#fff",
                        pointBackgroundColor:pointcolor,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                        data: final_data[i]});
        }
        var datatotal=[
                        {
                            fill: false,
                            cubicInterpolationMode:"monotone",
                            backgroundColor: "rgba(66,129,164,0.2)",
                            borderColor: "rgba(216, 212, 212, 0.5)",
                            pointBorderColor: "#fff",
                            pointBackgroundColor:pointcolor,
                            pointRadius: 4,
                            pointHoverRadius: 6,
                            data: datadown
                        }
                      ];
        for(var i=0;i<segment_data.length;i++){
            datatotal.push(present_data[i]);
        }
        datatotal.push({
                            fill: false,
                            cubicInterpolationMode:"monotone",
                            backgroundColor: "rgba(66,129,164,0.2)",
                            borderColor: "rgba(216, 212, 212, 0.5)",
                            pointBorderColor: "#fff",
                            pointBackgroundColor:pointcolor,
                            pointRadius: 4,
                            pointHoverRadius: 6,
                            data: dataup
                        });

        var chart = new Chart(chartRadarDOM, {
            type: 'line',
            data:{
                labels: labels,
                datasets:datatotal
            },
            options: chartRadarOptions
        });

        //click point handling
        canv.onclick = function(evt){
            var activepoints = chart.getElementAtEvent(evt);
            if (activepoints[0]){
                var id = set + '-' + (activepoints[0]['_index']+1)
                console.log(id)

                //init game
                if (!game_name){
                    game_name = "18IND_TC";
                }
                filename = 'statistics/rally_type_real_' + game_name + '.json';

                $.getJSON(filename, function(data2) {
                    document.getElementById("rallytitle").innerHTML = id + ' 球種分佈圖';
                    //filter data to specific set
                    data2 = data2.filter(function(item) {
                        return item.set == set
                    });
                    data2 = data2[0].info;

                    //get index from json file
                    index = data2.findIndex(function(item){
                        return id.split('-')[1] == item.rally;
                    });
    
                    var labels = data2.map(function(item) {
                        return item.result.map(function(e){
                            return e.balltype;            
                        })
                    });
    
                    var dataA = [];
                    for(var i = 0;i<data2[index].result.length;i++){
                        dataA.push(data2[index].result[i].count)
                    }
    
                    var dataB = [];
                    for(var i = 0;i<data2[index+1].result.length;i++){
                        dataB.push(data2[index+1].result[i].count)
                    }

                    console.log(dataA)
                    console.log(dataB)

                    $("#radarChart").show(function(event){
                        //filter data to specific rally
                        data_choose = data.filter(function(item) {
                            return item.rally == id.split('-')[1];
                        });
                        data_choose = data_choose[0];
                        // console.log(data_choose);
                        
                        // show win_reason and lose balltype on each rally
                        document.getElementById("lose_reason").innerHTML = '失分原因：' + data_choose.on_off_court;
                        document.getElementById("lose_balltype").innerHTML = '失分球種：' + data_choose.balltype;

                        var modal = $(this);
                        var canvas = modal.find('.modal-body canvas');
                        var ctx = canvas[0].getContext("2d"); 
                        var chart = new Chart(ctx, {
                            type: "radar",
                            data: {
                                labels: labels[0],
                                datasets: [
                                    {
                                    label: "Player A",
                                    fill: true,
                                    backgroundColor: "rgba(66,129,164,0.2)",
                                    borderColor: "rgba(66,129,164,0.8)",
                                    pointBorderColor: "#fff",
                                    pointBackgroundColor: "rgba(66,129,164,1)",
                                    data: dataA
                                    }, {
                                    label: "Player B",
                                    fill: true,
                                    backgroundColor: "rgba(255,99,132,0.2)",
                                    borderColor: "rgba(255,99,132,1)",
                                    pointBorderColor: "#fff",
                                    pointBackgroundColor: "rgba(255,99,132,1)",
                                    pointBorderColor: "#fff",
                                    data: dataB
                                    }
                                ]
                            },
                            options: {
                                scale:{
                                    ticks:{
                                        min:0,
                                        stepSize:1
                                    },
                                    pointLabels: { 
                                        fontSize:14 
                                    }
                                },
                                legend:{
                                    labels:{
                                        fontColor: 'rgba(248, 184, 82, 1)',
                                        fontSize: 16
                                    }
                                }
                            }
                        });
                    });

                    //close modal chart
                    $(function() {
                        $('.close').click(function() {
                            $('#radarChart').hide(function(event){
                                var modal = $(this);
                                var canvas = modal.find('.modal-body canvas');
                                var ctx = canvas[0].getContext("2d"); 
                                $(".modal-body canvas").remove();
                                $(".modal-body").html('<canvas id="canvas" width="1000" height="800"></canvas>\
                                                    <div class="modal-text" id="lose_reason"></div>\
                                                    <div class="modal-text" id="lose_balltype"></div>');
                            });
                        });
                    });
                })
            }
            
        }
    });
}

function init_on_off_court(minrally,maxrally,set,game_name){
    //create player info radar
    $('#on_off_court .playerA').html('<div class="subtitle">選手A失分比例</div>\
    <canvas id="on_off_court_chartA" width="800" height="600"></canvas>');
    $('#on_off_court .playerB').html('<div class="subtitle">選手B失分比例</div>\
    <canvas id="on_off_court_chartB" width="800" height="600"></canvas>');  

    var chartRadarDOMA;
    var chartRadarDOMB;
    var chartRadarOptions;

    // Chart.defaults.global.responsive = false;
    chartRadarDOMA = document.getElementById("on_off_court_chartA");
    chartRadarDOMB = document.getElementById("on_off_court_chartB");
    //custormized options
    chartRadarOptions = 
    {
        legend:{
            labels:{
                fontColor: 'rgba(248, 184, 82, 1)',
                fontSize: 16,
                fontStyle: "bold"
            }
        }
        // responsive:false
    };

    //init game
    if (!game_name){
        game_name = "18IND_TC";
    }
    filename = 'statistics/rally_count_real_' + game_name + '.json';

    $.getJSON(filename, function(data) {
        [data,set,minrally,maxrally] = data_filter(data,set,minrally,maxrally,1);

        //filter winners
        dataB = data.filter(function(item){
            return item.winner == 'A'
        });
        dataA = data.filter(function(item){
            return item.winner == 'B'
        });
        console.log("dataA = ",dataA.length);
        console.log("dataB = ",dataB.length);
        
        //count each reason
        var group_data = Object.keys(_.groupBy(data,"on_off_court"));
        var sum_dataA = new Array(group_data.length).fill(0);
        var sum_dataB = new Array(group_data.length).fill(0);
        for(var i = 0;i<dataA.length;i++){
            for(var j = 0;j<group_data.length;j++){
                if (dataA[i].on_off_court == group_data[j]){
                    sum_dataA[j] +=1;
                }
            }
        }
        for(var i = 0;i<dataB.length;i++){
            for(var j = 0;j<group_data.length;j++){
                if (dataB[i].on_off_court == group_data[j])
                    sum_dataB[j] +=1;
            }
        }

        console.log(sum_dataA);
        console.log(sum_dataB);
        
        var labels = group_data;

        //random color generator
        color = new Array();
        for(var i = 0;i<data.length;i++){
            r = Math.floor(Math.random() * 256);
            g = Math.floor(Math.random() * 256);
            b = Math.floor(Math.random() * 256);
            color.push('rgb(' + r + ', ' + g + ', ' + b + ')');
        }
        
        var chart = new Chart(chartRadarDOMA, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    backgroundColor: color,
                    pointBorderColor: "rgba(0,0,0,0)",
                    borderColor: 'rgb(17, 16, 17)',
                    borderWidth: 1,
                    data: sum_dataA
                }]
            },
            options: chartRadarOptions
        });

        var chart = new Chart(chartRadarDOMB, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    backgroundColor: color,
                    pointBorderColor: "rgba(0,0,0,0)",
                    borderColor: 'rgb(17, 16, 17)',
                    borderWidth: 1,
                    data: sum_dataB
                }]
            },
            options: chartRadarOptions
        });
    });
}

function init_lose_error(minrally,maxrally,set,game_name){
    //create player info radar
    $('#lose_error .playerA').html('<div class="subtitle">選手A失誤比例</div>\
    <canvas id="lose_error_chartA" width="800" height="600"></canvas>');
    $('#lose_error .playerB').html('<div class="subtitle">選手B失誤比例</div>\
    <canvas id="lose_error_chartB" width="800" height="600"></canvas>');  

    var chartRadarDOMA;
    var chartRadarDOMB;
    var chartRadarOptions;

    // Chart.defaults.global.responsive = false;
    chartRadarDOMA = document.getElementById("lose_error_chartA");
    chartRadarDOMB = document.getElementById("lose_error_chartB");
    //custormized options
    chartRadarOptions = 
    {
        legend:{
            labels:{
                fontColor: 'rgba(248, 184, 82, 1)',
                fontSize: 16,
                fontStyle: "bold"
            }
        }
        // responsive:false
    };

    //init game
    if (!game_name){
        game_name = "18IND_TC";
    }
    filename = 'statistics/rally_count_real_' + game_name + '.json';

    $.getJSON(filename, function(data) {
        [data,set,minrally,maxrally] = data_filter(data,set,minrally,maxrally,1);

        //filter winners
        dataB = data.filter(function(item){
            return item.winner == 'A'
        });
        dataA = data.filter(function(item){
            return item.winner == 'B'
        });
        console.log("dataA = ",dataA.length);
        console.log("dataB = ",dataB.length);
        
        //count each reason
        var group_data = Object.keys(_.groupBy(data,"error"));
        var sum_dataA = new Array(group_data.length).fill(0);
        var sum_dataB = new Array(group_data.length).fill(0);
        for(var i = 0;i<dataA.length;i++){
            for(var j = 0;j<group_data.length;j++){
                if (dataA[i].error == group_data[j]){
                    sum_dataA[j] +=1;
                }
            }
        }
        for(var i = 0;i<dataB.length;i++){
            for(var j = 0;j<group_data.length;j++){
                if (dataB[i].error == group_data[j])
                    sum_dataB[j] +=1;
            }
        }

        console.log(sum_dataA);
        console.log(sum_dataB);
        
        var labels = ["受迫性失誤","非受迫性失誤"];

        //random color generator
        color = new Array();
        for(var i = 0;i<data.length;i++){
            r = Math.floor(Math.random() * 256);
            g = Math.floor(Math.random() * 256);
            b = Math.floor(Math.random() * 256);
            color.push('rgb(' + r + ', ' + g + ', ' + b + ')');
        }
        
        var chart = new Chart(chartRadarDOMA, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    backgroundColor: color,
                    pointBorderColor: "rgba(0,0,0,0)",
                    borderColor: 'rgb(17, 16, 17)',
                    borderWidth: 1,
                    data: sum_dataA
                }]
            },
            options: chartRadarOptions
        });

        var chart = new Chart(chartRadarDOMB, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    backgroundColor: color,
                    pointBorderColor: "rgba(0,0,0,0)",
                    borderColor: 'rgb(17, 16, 17)',
                    borderWidth: 1,
                    data: sum_dataB
                }]
            },
            options: chartRadarOptions
        });
    });
}

function init_total_balltype(minrally,maxrally,set,game_name){
    $('#total_balltype .playerA').html('<div class="subtitle">選手A獲勝球種</div>\
    <canvas id="total_balltype_chartA" width="800" height="600"></canvas>');
    $('#total_balltype .playerB').html('<div class="subtitle">選手B獲勝球種</div>\
    <canvas id="total_balltype_chartB" width="800" height="600"></canvas>');

    $('#sum_balltype .playerA').html('<div class="subtitle">選手A球種統計</div>\
    <canvas id="sum_balltype_chartA" width="800" height="600"></canvas>');
    $('#sum_balltype .playerB').html('<div class="subtitle">選手B球種統計</div>\
    <canvas id="sum_balltype_chartB" width="800" height="600"></canvas>');

    //init game
    if (!game_name){
        game_name = "18IND_TC";
    }
    filename = 'statistics/rally_type_real_' + game_name + '.json';

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

        // init minrally and maxrally if are undefined,null,0,NaN,empty string,false
        if (!minrally){
            minrally = Math.min.apply(Math, data.map(function(d) { 
                return d.rally; 
            }));
        }
        if (!maxrally){
            maxrally = Math.max.apply(Math, data.map(function(d) { 
                return d.rally; 
            }));
        }

        //filter data to specific interval
        data = data.filter(function(item) {
            return parseInt(item.rally) >= minrally && parseInt(item.rally) <= maxrally;
        });

        // console.log(set);
        // console.log(minrally);
        // console.log(maxrally);

        var labels = data.map(function(item) {
            return item.result.map(function(e){
                return e.balltype;            
            })
        });

        var total = data.map(function(item){
            return item.result
        });

        //custormized options
        var chartRadarOptions = 
        {
            scale:{
                ticks:{
                    min:0,
                    // stepSize:10
                },
                pointLabels: { 
                    fontSize:14 
                }
            },
            legend:{
                labels:{
                    fontColor: 'rgba(248, 184, 82, 1)',
                    fontSize: 16
                }
            }
        };

        filename = 'statistics/rally_count_real_' + game_name + '.json';

        //rendering each player win balltype
        $.getJSON(filename, function(data2) {
            [data2,set,minrally,maxrally] = data_filter(data2,set,minrally,maxrally,1);

            //filter winners
            data2A = data2.filter(function(item){
                return item.winner == 'A'
            });
            data2B = data2.filter(function(item){
                return item.winner == 'B'
            });

            var dataA = new Array(data[0].result.length).fill(0);
            var dataB = new Array(data[0].result.length).fill(0);

            for(var i = 0;i<data2A.length;i++){
                for(var j = 0;j<labels[0].length;j++){
                    if (data2A[i].balltype == labels[0][j])
                        dataA[j] += 1;
                }
            };

            for(var i = 0;i<data2B.length;i++){
                for(var j = 0;j<labels[0].length;j++){
                    if (data2B[i].balltype == labels[0][j])
                        dataB[j] += 1;
                }
            };

            // console.log(labels[0]);
            // console.log(dataA);
            // console.log(dataB);

            var chartRadarOptionsPlayer = 
            {
                scale:{
                    ticks:{
                        min:0,
                        stepSize:1
                    },
                    pointLabels: { 
                        fontSize:14 
                    }
                },
                legend:{
                    labels:{
                        fontColor: 'rgba(248, 184, 82, 1)',
                        fontSize: 16
                    }
                }
            };

            //create player info radar            
            chartRadarDOMA = document.getElementById("total_balltype_chartA");
            var chart = new Chart(chartRadarDOMA, {
                type: 'radar',
                data:{
                    labels: labels[0],
                    datasets: [
                        {
                        label: "Player A",
                        fill: true,
                        cubicInterpolationMode:"monotone",
                        backgroundColor: "rgba(66,129,164,0.2)",
                        borderColor: "rgba(66,129,164,1)",
                        pointBorderColor: "#fff",
                        pointBackgroundColor: "rgba(66,129,164,1)",
                        data: dataA
                        }
                    ]
                },
                options: chartRadarOptionsPlayer
            });

            //rendering winner B balltype

            chartRadarDOMB = document.getElementById("total_balltype_chartB");
            var chart = new Chart(chartRadarDOMB, {
                type: 'radar',
                data:{
                    labels: labels[0],
                    datasets: [{
                        label: "Player B",
                        fill: true,
                        cubicInterpolationMode:"monotone",
                        backgroundColor: "rgba(255,99,132,0.2)",
                        borderColor: "rgba(255,99,132,1)",
                        pointBorderColor: "#fff",
                        pointBackgroundColor: "rgba(255,99,132,1)",
                        pointBorderColor: "#fff",
                        data: dataB
                        }
                    ]
                },
                options: chartRadarOptionsPlayer
            });
        })
        .done(function(){
            var dataA = new Array(data[0].result.length).fill(0);
            var dataB = new Array(data[0].result.length).fill(0);
            for(var i = 0;i<data.length;i+=2){
                rally = parseInt(data[i].rally);
                for(var j = 0;j<data[i].result.length;j++){
                    dataA[j] += data[i].result[j].count;
                    dataB[j] += data[i+1].result[j].count;
                }
            };
            console.log(dataA)
            console.log(dataB)
            //rendering total balltype
            var chartRadarDOM = document.getElementById("sum_balltype_chartA");
            var chart = new Chart(chartRadarDOM, {
                type: 'radar',
                data:{
                    labels: labels[0],
                    datasets: [
                        {
                        label: "Player A",
                        fill: true,
                        cubicInterpolationMode:"monotone",
                        backgroundColor: "rgba(66,129,164,0.2)",
                        borderColor: "rgba(66,129,164,1)",
                        pointBorderColor: "#fff",
                        pointBackgroundColor: "rgba(66,129,164,1)",
                        data: dataA
                        }
                    ]
                },
                options: chartRadarOptions
            });

            var chartRadarDOM = document.getElementById("sum_balltype_chartB");
            var chart = new Chart(chartRadarDOM, {
                type: 'radar',
                data:{
                    labels: labels[0],
                    datasets: [
                        {
                        label: "Player B",
                        fill: true,
                        cubicInterpolationMode:"monotone",
                        backgroundColor: "rgba(255,99,132,0.2)",
                        borderColor: "rgba(255,99,132,1)",
                        pointBorderColor: "#fff",
                        pointBackgroundColor: "rgba(255,99,132,1)",
                        pointBorderColor: "#fff",
                        data: dataB
                        }
                    ]
                },
                options: chartRadarOptions
            });
        });
    });
}

function init_stroke_distribution(minrally,maxrally,set,game_name){
    //create player info radar
    $('#stroke_distribution .playerA').html('<div class="subtitle">選手A得分拍數</div>\
    <canvas id="stroke_distribution_chartA" width="800" height="600"></canvas>');
    $('#stroke_distribution .playerB').html('<div class="subtitle">選手B得分拍數</div>\
    <canvas id="stroke_distribution_chartB" width="800" height="600"></canvas>');

    var chartRadarDOMA;
    var chartRadarDOMB;
    var chartRadarOptions;

    // Chart.defaults.global.responsive = false;
    chartRadarDOMA = document.getElementById("stroke_distribution_chartA");
    chartRadarDOMB = document.getElementById("stroke_distribution_chartB");
    //custormized options
    chartRadarOptions = 
    {
        legend:{
            labels:{
                fontColor: 'rgba(248, 184, 82, 1)',
                fontSize: 16,
                fontStyle: "bold"
            }
        }
    };

    //init game
    if (!game_name){
        game_name = "18IND_TC";
    }
    filename = 'statistics/rally_count_real_' + game_name + '.json';

    $.getJSON(filename, function(data) {
        [data,set,minrally,maxrally] = data_filter(data,set,minrally,maxrally,1);

        //filter winners
        dataA = data.filter(function(item){
            return item.winner == 'A'
        });
        dataB = data.filter(function(item){
            return item.winner == 'B'
        });

        console.log(set);
        console.log(minrally);
        console.log(maxrally);
        var maxA=0;
        for(var i=0;i<dataA.length;i++)
        {
            if(dataA[i].stroke>maxA)
            {
                maxA=dataA[i].stroke;
            }
        }
        var maxB=0;
        for(var i=0;i<dataB.length;i++)
        {
            if(dataB[i].stroke>maxB)
            {
                maxB=dataB[i].stroke;
            }
        }
        var max_all=maxA>maxB?maxA:maxB;
        var labels = new Array(Math.ceil(max_all/5)).fill('');
        for(var i=0;i<Math.ceil(max_all/5);i++)
        {
            labels[i]=i*5+1+'~'+(i+1)*5;
        }
        var sum_dataA = new Array(Math.ceil(max_all/5)).fill(0);
        var sum_dataB = new Array(Math.ceil(max_all/5)).fill(0);
        for(var i = 0;i<dataA.length;i++){
            sum_dataA[Math.ceil(dataA[i].stroke/5)-1]+=1
        }
        for(var i = 0;i<dataB.length;i++){
            sum_dataB[Math.ceil(dataB[i].stroke/5)-1]+=1
        }
        for(var i = 0;i<sum_dataA.length;i++){
            if(sum_dataA[i]==0){
                sum_dataA[i]=NaN;
            }
        }
        for(var i = 0;i<sum_dataB.length;i++){
            if(sum_dataB[i]==0){
                sum_dataB[i]=NaN;
            }
        }
        console.log(sum_dataA);
        console.log(sum_dataB);


        //random color generator
        color = new Array();
        for(var i = 0;i<data.length;i++){
            r = Math.floor(Math.random() * 256);
            g = Math.floor(Math.random() * 256);
            b = Math.floor(Math.random() * 256);
            color.push('rgb(' + r + ', ' + g + ', ' + b + ')');
        }

        var chart = new Chart(chartRadarDOMA, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    backgroundColor: color,
                    pointBorderColor: "rgba(0,0,0,0)",
                    borderColor: 'rgb(17, 16, 17)',
                    borderWidth: 1,
                    data: sum_dataA
                }]
            },
            options: chartRadarOptions
        });

        var chart = new Chart(chartRadarDOMB, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    backgroundColor: color,
                    pointBorderColor: "rgba(0,0,0,0)",
                    borderColor: 'rgb(17, 16, 17)',
                    borderWidth: 1,
                    data: sum_dataB
                }]
            },
            options: chartRadarOptions
        });
    });
}

function init_court_distribution(minrally,maxrally,set,game_name){
    //左邊是後場 右邊是前場 小戴是A 陳是B
    $('#total_court .playerA').html('<div class="subtitle">選手A失分分佈</div>\
    <img id="badminton_courtA" src="../image/badminton_court.jpg" width="100%"/>');
    $('#total_court .playerB').html('<div class="subtitle">選手B失分分佈</div>\
    <img id="badminton_courtB" src="../image/badminton_court.jpg" class="img-fluid" width="100%"/>');

    var oriw = 930;
    var orih = 450;
    var imgA = document.getElementById("badminton_courtA");
    var canv = document.createElement('canvas');
    canv.id = 'total_court_chartA';
    canv.width = imgA.width;
    canv.height = orih/oriw*imgA.width;         //ori height is 450,ori width is 930,need get ori img size
    document.getElementById("total_court").getElementsByClassName("playerA")[0].appendChild(canv);
    var canvA = document.getElementById("total_court_chartA");
    canvA.style.position = "absolute";
    canvA.style.left = imgA.offsetLeft + 'px';
    canvA.style.top = imgA.Top + 'px';

    var imgB = document.getElementById("badminton_courtB");
    var canv = document.createElement('canvas');
    canv.id = 'total_court_chartB';
    canv.width = imgB.width;
    canv.height = orih/oriw*imgB.width;
    document.getElementById("total_court").getElementsByClassName("playerB")[0].appendChild(canv);
    var canvB = document.getElementById("total_court_chartB");
    canvB.style.position = "absolute";
    canvB.style.left = imgB.offsetLeft + 'px';
    canvB.style.top = imgB.Top + 'px';

    var ctxA = canvA.getContext("2d");
    var ctxB = canvB.getContext("2d");

    //init game
    if (!game_name){
        game_name = "18IND_TC";
    }
    filename = 'statistics/rally_count_real_' + game_name + '.json';

    $.getJSON(filename, function(data) {
        [data,set,minrally,maxrally] = data_filter(data,set,minrally,maxrally,1);

        //filter winners
        dataB = data.filter(function(item){
            return item.winner == 'A'
        });
        dataA = data.filter(function(item){
            return item.winner == 'B'
        });

        //count each area
        var group_data = Object.keys(_.groupBy(data,"lose_area")).sort();
        var sum_dataA = new Object();
        sum_dataA.area = group_data;
        sum_dataA.value = new Array(group_data.length).fill(0);
        sum_dataA.selfout = new Array(group_data.length).fill(0);
        var sum_dataB = new Object();
        sum_dataB.area = group_data;
        sum_dataB.value = new Array(group_data.length).fill(0);
        sum_dataB.selfout = new Array(group_data.length).fill(0);

        var sumA = 0;
        for(var i = 0;i<dataA.length;i++){
            for(var j = 0;j<group_data.length;j++){
                if (dataA[i].lose_area == group_data[j]){
                    if ((dataA[i].lose_area.split('')[0] == 'F' || dataA[i].lose_area.split('')[0] == 'E' || dataA[i].lose_area.split('')[1] == '5') && dataA[i].on_off_court == '未回擊成功'){
                        sum_dataA.selfout[j] +=1;
                    }
                    else{
                        sum_dataA.value[j] +=1;
                    }
                    sumA++;
                }
            }
        }
        var sumB = 0;
        for(var i = 0;i<dataB.length;i++){
            for(var j = 0;j<group_data.length;j++){
                if (dataB[i].lose_area == group_data[j]){
                    if ((dataB[i].lose_area.split('')[0] == 'F' || dataB[i].lose_area.split('')[0] == 'E' || dataB[i].lose_area.split('')[1] == '5') && dataB[i].on_off_court == '未回擊成功'){
                        sum_dataB.selfout[j] +=1;
                    }
                    else{
                        sum_dataB.value[j] +=1;
                    }
                    sumB++;
                }
            }
        }

        var court = new Object();
        var coord_orix = 935;
        var coord_oriy = 424;
        var brutew = 61/oriw*canvA.width;
        var bruteh = 41/orih*canvA.height;
        court.xarea = ['1','2','3','4'];
        court.yarea = ['A','B','C','D','E'];
        court.xcoord_back = [[307,468],[74,307],[0,74],[-60,0]];
        court.xcoord_back = court.xcoord_back.map(function(item){
            return [parseInt(item[0]/coord_orix*(canvA.width-2*brutew)+brutew),parseInt(item[1]/coord_orix*(canvA.width-2*brutew)+brutew)];
        });
        court.ycoord_back = [[108,316],[316,392],[32,108],[392,500],[-50,32]];
        court.ycoord_back = court.ycoord_back.map(function(item){
            return [parseInt(item[0]/coord_oriy*(canvA.height-2*bruteh)+bruteh),parseInt(item[1]/coord_oriy*(canvA.height-2*bruteh)+bruteh)];
        });

        court.xcoord_front = [[468,629],[629,861],[861,935],[935,995]];
        court.xcoord_front = court.xcoord_front.map(function(item){
            return [parseInt(item[0]/coord_orix*(canvB.width-2*brutew)+brutew),parseInt(item[1]/coord_orix*(canvB.width-2*brutew)+brutew)];
        });
        court.ycoord_front = [[108,316],[32,108],[316,392],[-50,32],[392,500]];
        court.ycoord_front = court.ycoord_front.map(function(item){
            return [parseInt(item[0]/coord_oriy*(canvB.height-2*bruteh)+bruteh),parseInt(item[1]/coord_oriy*(canvB.height-2*bruteh)+bruteh)];
        });

        // console.log(canvA.width);
        // console.log(canvA.height);
        // console.log(court);
        console.log(sum_dataA);
        console.log(sum_dataB);

        //render middle line on courtA and courtB
        ctxA.fillStyle = "rgb(0,0,0)";
        ctxA.fillRect(court.xcoord_back[0][1]-2,court.ycoord_back[4][0],4,court.ycoord_back[3][1] - court.ycoord_back[4][0]);
        ctxB.fillStyle = "rgb(0,0,0)";
        ctxB.fillRect(court.xcoord_back[0][1]-2,court.ycoord_back[4][0],4,court.ycoord_back[3][1] - court.ycoord_back[4][0]);

        //render rectangle ratio area over image
        for(var i = 0;i<sum_dataA.value.length;i++){
            if (sum_dataA.value[i] != 0){
                var ratio = (sum_dataA.value[i]/sumA).toFixed(2);
                var idx = court.xarea.indexOf(sum_dataA.area[i].split('')[1]);
                var idy = court.yarea.indexOf(sum_dataA.area[i].split('')[0]);
                var topX,topY,w,h;

                //out-field ball will on opposite field
                if (set%2 == 1){
                    if (sum_dataA.area[i].split('')[0] == 'D' || sum_dataA.area[i].split('')[0] == 'E' || sum_dataA.area[i].split('')[1] == '4'){
                        topX = court.xcoord_front[idx][0];
                        topY = court.ycoord_front[idy][0];
                        w = court.xcoord_front[idx][1]-court.xcoord_front[idx][0];
                        h = court.ycoord_front[idy][1]-court.ycoord_front[idy][0];
                    }
                    else{
                        topX = court.xcoord_back[idx][0];
                        topY = court.ycoord_back[idy][0];
                        w = court.xcoord_back[idx][1]-court.xcoord_back[idx][0];
                        h = court.ycoord_back[idy][1]-court.ycoord_back[idy][0];
                    }
                }
                else{
                    if (sum_dataA.area[i].split('')[0] == 'D' || sum_dataA.area[i].split('')[0] == 'E' || sum_dataA.area[i].split('')[1] == '4'){
                        topX = court.xcoord_back[idx][0];
                        topY = court.ycoord_back[idy][0];
                        w = court.xcoord_back[idx][1]-court.xcoord_back[idx][0];
                        h = court.ycoord_back[idy][1]-court.ycoord_back[idy][0];
                    }
                    else{
                        topX = court.xcoord_front[idx][0];
                        topY = court.ycoord_front[idy][0];
                        w = court.xcoord_front[idx][1]-court.xcoord_front[idx][0];
                        h = court.ycoord_front[idy][1]-court.ycoord_front[idy][0];
                    }
                }
                ctxA.fillStyle = "rgba(66,129,164," + ratio + ")";
                ctxA.fillRect(topX,topY,w,h);
                ctxA.fillStyle = "rgb(0,0,0)";
                ctxA.textAlign = "center"; 
                ctxA.textBaseline = "middle";
                ctxA.strokeText(ratio,topX+w/2,topY+h/2);
            }
            if (sum_dataA.selfout[i] != 0){
                var ratio = (sum_dataA.selfout[i]/sumA).toFixed(2);
                var idx = court.xarea.indexOf(sum_dataA.area[i].split('')[1]);
                var idy = court.yarea.indexOf(sum_dataA.area[i].split('')[0]);
                var topX,topY,w,h;
                if (set%2 == 1){
                    topX = court.xcoord_back[idx][0];
                    topY = court.ycoord_back[idy][0];
                    w = court.xcoord_back[idx][1]-court.xcoord_back[idx][0];
                    h = court.ycoord_back[idy][1]-court.ycoord_back[idy][0];
                }
                else{
                    topX = court.xcoord_front[idx][0];
                    topY = court.ycoord_front[idy][0];
                    w = court.xcoord_front[idx][1]-court.xcoord_front[idx][0];
                    h = court.ycoord_front[idy][1]-court.ycoord_front[idy][0];
                }
                ctxA.fillStyle = "rgba(66,129,164," + ratio + ")";
                ctxA.fillRect(topX,topY,w,h);
                ctxA.fillStyle = "rgb(0,0,0)";
                ctxA.textAlign = "center"; 
                ctxA.textBaseline = "middle";
                ctxA.strokeText(ratio,topX+w/2,topY+h/2);
            }
        }

        for(var i = 0;i<sum_dataB.value.length;i++){
            if (sum_dataB.value[i] != 0){
                var ratio = (sum_dataB.value[i]/sumB).toFixed(2);
                var idx = court.xarea.indexOf(sum_dataB.area[i].split('')[1]);
                var idy = court.yarea.indexOf(sum_dataB.area[i].split('')[0]);
                var topX,topY,w,h;
                if(set%2 == 1){
                    if (sum_dataB.area[i].split('')[0] == 'D' || sum_dataB.area[i].split('')[0] == 'E' || sum_dataB.area[i].split('')[1] == '4'){
                        topX = court.xcoord_back[idx][0];
                        topY = court.ycoord_back[idy][0];
                        w = court.xcoord_back[idx][1]-court.xcoord_back[idx][0];
                        h = court.ycoord_back[idy][1]-court.ycoord_back[idy][0];
                    }
                    else{
                        topX = court.xcoord_front[idx][0];
                        topY = court.ycoord_front[idy][0];
                        w = court.xcoord_front[idx][1]-court.xcoord_front[idx][0];
                        h = court.ycoord_front[idy][1]-court.ycoord_front[idy][0];
                    }
                }
                else{
                    if (sum_dataB.area[i].split('')[0] == 'D' || sum_dataB.area[i].split('')[0] == 'E' || sum_dataB.area[i].split('')[1] == '4'){
                        topX = court.xcoord_front[idx][0];
                        topY = court.ycoord_front[idy][0];
                        w = court.xcoord_front[idx][1]-court.xcoord_front[idx][0];
                        h = court.ycoord_front[idy][1]-court.ycoord_front[idy][0];
                    }
                    else{
                        topX = court.xcoord_back[idx][0];
                        topY = court.ycoord_back[idy][0];
                        w = court.xcoord_back[idx][1]-court.xcoord_back[idx][0];
                        h = court.ycoord_back[idy][1]-court.ycoord_back[idy][0];
                    }
                }
                ctxB.fillStyle = "rgba(255,99,132," + ratio + ")";
                ctxB.fillRect(topX,topY,w,h);
                ctxB.fillStyle = "rgb(0,0,0)";
                ctxB.textAlign = "center"; 
                ctxB.textBaseline = "middle";
                ctxB.strokeText(ratio,topX+w/2,topY+h/2);
            }
            if (sum_dataB.selfout[i] != 0){
                var ratio = (sum_dataB.selfout[i]/sumB).toFixed(2);
                var idx = court.xarea.indexOf(sum_dataB.area[i].split('')[1]);
                var idy = court.yarea.indexOf(sum_dataB.area[i].split('')[0]);
                var topX,topY,w,h;
                if (set%2 == 1){
                    topX = court.xcoord_front[idx][0];
                    topY = court.ycoord_front[idy][0];
                    w = court.xcoord_front[idx][1]-court.xcoord_front[idx][0];
                    h = court.ycoord_front[idy][1]-court.ycoord_front[idy][0];
                }
                else{
                    topX = court.xcoord_back[idx][0];
                    topY = court.ycoord_back[idy][0];
                    w = court.xcoord_back[idx][1]-court.xcoord_back[idx][0];
                    h = court.ycoord_back[idy][1]-court.ycoord_back[idy][0];
                }
                ctxA.fillStyle = "rgba(66,129,164," + ratio + ")";
                ctxA.fillRect(topX,topY,w,h);
                ctxA.fillStyle = "rgb(0,0,0)";
                ctxA.textAlign = "center"; 
                ctxA.textBaseline = "middle";
                ctxA.strokeText(ratio,topX+w/2,topY+h/2);
            }
        }
    })
}

function change_interval(){
    //get interval when clicking submit
    var minrally = document.getElementById("down").value;
    var maxrally = document.getElementById("up").value;
    var set = document.getElementById("set").value;
    var game = document.getElementById("game").value;

    //delete old linechart
    $('#line_chart').remove();
    init_linechart(minrally,maxrally,set,game);

    //delete old doughnut
    $('#on_off_court .subtitle').remove();
    $('#on_off_court_chartA').remove();
    $('#on_off_court_chartB').remove();
    init_on_off_court(minrally,maxrally,set,game);

    //delete old error doughnut
    $('#lose_error .subtitle').remove();
    $('#lose_error_chartA').remove();
    $('#lose_error_chartB').remove();
    init_lose_error(minrally,maxrally,set,game);

    //delete old radar
    $('#total_balltype .subtitle').remove();
    $('#total_balltype_chartA').remove();
    $('#total_balltype_chartB').remove();
    $('#sum_balltype_chartA').remove();
    $('#sum_balltype_chartB').remove();
    init_total_balltype(minrally,maxrally,set,game);

    //delete old stoke distribution
    $('#stroke_distribution .subtitle').remove();
    $('#stroke_distribution_chartA').remove();
    $('#stroke_distribution_chartB').remove();
    init_stroke_distribution(minrally,maxrally,set,game);

    //delete old court distribution
    $('#total_court .subtitle').remove();
    $('#total_court_chartA').remove();
    $('#total_court_chartB').remove();
    $('#badminton_courtA').remove();
    $('#badminton_courtB').remove();
    init_court_distribution(minrally,maxrally,set,game);
}

function change_set() {
    game = document.getElementById("game").value;
    new_set = document.getElementById("set").value;
    $('#down option').remove();
    $('#up option').remove();
    get_interval_updown(new_set,game);

    //delete old and refresh new linechart
    $('#line_chart').remove();
    init_linechart(null,null,new_set,game);

    //delete old doughnut
    $('#on_off_court .subtitle').remove();
    $('#on_off_court_chartA').remove();
    $('#on_off_court_chartB').remove();
    init_on_off_court(null,null,new_set,game);

    //delete old error doughnut
    $('#lose_error .subtitle').remove();
    $('#lose_error_chartA').remove();
    $('#lose_error_chartB').remove();
    init_lose_error(null,null,new_set,game);

    //delete old radar
    $('#total_balltype .subtitle').remove();
    $('#total_balltype_chartA').remove();
    $('#total_balltype_chartB').remove();
    $('#sum_balltype_chartA').remove();
    $('#sum_balltype_chartB').remove();
    init_total_balltype(null,null,new_set,game);

    //delete old stoke distribution
    $('#stroke_distribution .subtitle').remove();
    $('#stroke_distribution_chartA').remove();
    $('#stroke_distribution_chartB').remove();
    init_stroke_distribution(null,null,new_set,game);

    //delete old court distribution
    $('#total_court .subtitle').remove();
    $('#total_court_chartA').remove();
    $('#total_court_chartB').remove();
    $('#badminton_courtA').remove();
    $('#badminton_courtB').remove();
    init_court_distribution(null,null,new_set,game);
}

function change_game() {
    new_game = document.getElementById("game").value;
    $('#set option').remove();
    get_interval_set();
    change_set();
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

function get_interval_set(){
    var game_name = document.getElementById("game").value;
    //init game
    if (!game_name){
        game_name = "18IND_TC";
    }
    filename = 'statistics/rally_count_real_' + game_name + '.json';

    $.getJSON(filename, function(data) {
        //find max set
        var maximum = 0;
        for (var i=0 ; i<data.length ; i++) {
            if (data[i].set > maximum)
                maximum = data[i].set;
        }

        for(var i=1;i<=maximum;i+=1)
        {
            var insertText = '<option value='+i+'>'+i+'</option>';
            $('#set').append(insertText); 
        }
    });
}

function get_interval_updown(set,game){
    var game_name = document.getElementById("game").value;
    //init game
    if (!game_name){
        game_name = "18IND_TC";
    }
    filename = 'statistics/rally_count_real_' + game_name + '.json';

    $.getJSON(filename, function(data) {
        //init set
        if (!set){
            set = 1;
        }

        //filter data to specific set
        data = data.filter(function(item) {
            return item.set == set
        });
        data = data[0].result
        maximum = Math.max.apply(Math, data.map(function(d) { 
            return d.rally;
        }));  
        for(var i=0;i<maximum;i+=1)
        {
            var score = data[i].score;
            var insertText = '<option value=' + (i+1) + '>' + score + ' (' + (i+1) + ')' + '</option>';
            $('#down').append(insertText); 
            $('#up').append(insertText); 
        }
        $('#up').val(maximum); 
    })
}

function data_filter(data,set,minrally,maxrally,mode){
    //init set
    if (!set){
        set = 1;
    }

    //filter data to specific set
    data = data.filter(function(item) {
        return item.set == set
    });
    data = data[0].result;

    // init minrally and maxrally if are undefined,null,0,NaN,empty string,false
    if (!minrally){
        minrally = Math.min.apply(Math, data.map(function(d) { 
            return d.rally; 
        }));
    }
    if (!maxrally){
        maxrally = Math.max.apply(Math, data.map(function(d) { 
            return d.rally; 
        }));
    }

    if(mode == 1){
        //filter data to specific interval
        data = data.filter(function(item) {
            return item.rally >= minrally && item.rally <= maxrally
        });
    }

    return [data,set,minrally,maxrally];
}