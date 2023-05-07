const express = require("express");
const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const { recommender } = require("./recommender");
const { model } = require('./model');
const app = express();
const PORT = 3000;

app.get("/", (req, res) => {

    // 𝗶𝗻 𝗰𝗮𝘀𝗲 𝗼𝗳 𝗿𝗲𝘁𝗿𝗮𝗶𝗻𝗻𝗶𝗻𝗴 𝘁𝗵𝗲 𝗺𝗼𝗱𝗲𝗹
    // model();

    //* 𝗚𝗲𝘁 𝘁𝗵𝗲 𝘁𝗿𝗮𝗶𝗻𝗲𝗱 𝘂𝘀𝗲𝗿 𝗮𝗻𝗱 𝗶𝘁𝗲𝗺 𝗺𝗮𝘁𝗿𝗶𝗰𝗲𝘀
    const loadedModel = JSON.parse(fs.readFileSync('model.json'));
    const userMatrix = tf.tensor(loadedModel.userMatrixData);
    const itemMatrix = tf.tensor(loadedModel.itemMatrixData);

    //* 𝗿𝗮𝘁𝗶𝗻𝗴𝘀𝗗𝗮𝘁𝗮 𝗳𝗿𝗼𝗺 𝗺𝘆 𝗗𝗮𝘁𝗮𝗯𝗮𝘀𝗲
    var matrix=[
        [0,2,0,1,3,0,0,0,0,5,0,2,0,0,4,0,0,0,0,0],
        [1,0,0,0,3,0,2,0,0,5,0,2,0,0,4,0,0,1,0,0],
        [1,0,3,4,0,2,0,0,0,0,1,2,0,0,4,0,0,0,2,0],
        [0,2,0,1,3,0,2,0,0,0,0,2,0,0,4,0,3,0,0,0],
        [0,2,0,1,0,0,3,0,0,0,0,2,0,0,4,0,2,0,0,5],
        [3,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,5],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ]

    //* 𝘁𝘂𝗿𝗻𝗶𝗻𝗴 𝘁𝗵𝗲 𝗺𝗮𝘁𝗿𝗶𝘅 𝗶𝗻𝘁𝗼 𝘁𝗲𝗻𝘀𝗼𝗿
    const ratingsData = tf.tensor2d(matrix);

    //* 𝘁𝗵𝗲 𝗿𝗮𝘁𝗶𝗻𝗴𝘀 𝘃𝗲𝗰𝘁𝗼𝗿 𝗳𝗼𝗿 𝘁𝗵𝗲 𝘂𝘀𝗲𝗿 𝗶 𝘄𝗮𝗻𝘁 𝘁𝗼 𝗽𝗿𝗲𝗱𝗶𝗰𝘁 𝗵𝗶𝘀 𝗿𝗮𝘁𝗶𝗻𝗴𝘀 𝗮𝗻𝗱 𝗿𝗲𝗰𝗼𝗺𝗺𝗲𝗻𝗱𝗶𝗻𝗴 𝘀𝗼𝗺𝗲 𝗽𝗿𝗱𝗼𝘂𝗰𝘁 𝗳𝗼𝗿 𝗵𝗶𝗺
    const userRatingsArray=matrix[0];

    //* 𝘁𝗵𝗲 𝗶𝘀𝗥𝗮𝘁𝗲𝗱 𝗶𝘀 𝘁𝗿𝘂𝗲 𝗶𝗳 𝘁𝗵𝗲 𝘂𝘀𝗲𝗿 𝗿𝗮𝘁𝗲𝗱 𝘀𝗼𝗺𝗲 𝗽𝗿𝗼𝗱𝘂𝗰𝘁𝘀
    var isRated=false;
    for (let i=0;i<userRatingsArray.length;i++){
        if(userRatingsArray[i]!=0){
            isRated=true;
            break;
        }
    }

    //* 𝗳𝗼𝗿 𝗿𝗲𝘁𝘂𝗿𝗻𝗶𝗻𝗴 𝘁𝗵𝗲 𝗶𝗱 𝗼𝗳 𝘁𝗵𝗲 𝗽𝗿𝗼𝗱𝘂𝗰𝘁 𝘄𝗵𝗶𝗰𝗵 𝗶 𝘄𝗶𝗹𝗹 𝗿𝗲𝗰𝗼𝗺𝗺𝗲𝗻𝗱 𝘁𝗼 𝘁𝗵𝗲 𝘂𝘀𝗲𝗿
    const itemId = [];
    for(let i=1;i<=ratingsData.shape[1];i++){
            itemId.push(`product ${i}`);
    }
    

    const r=recommender(itemId,10,ratingsData,isRated,userRatingsArray,itemMatrix);

    res.send(r);

});


app.listen(PORT, () => {
    console.log(`server run on http://localhost:${PORT}`);
});