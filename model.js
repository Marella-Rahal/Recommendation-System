const tf = require('@tensorflow/tfjs');
const fs = require('fs');

exports.model=()=>{

    // 𝗟𝗼𝗮𝗱 𝘁𝗵𝗲 𝗱𝗮𝘁𝗮𝘀𝗲𝘁 
    // Read the CSV file
    const csv = fs.readFileSync('readyDataSet.csv','utf-8');
    // Split the CSV into rows and columns
    const rows = csv.split('\n');
    const cols = rows.map(row => row.split(','));
    // Convert the values to numbers
    const matrix = cols.map(row => row.map(val => Number(val)));

    // 𝗗𝗲𝗳𝗶𝗻𝗲 𝘁𝗵𝗲 𝗶𝗻𝗽𝘂𝘁 𝗱𝗮𝘁𝗮 (𝘂𝘀𝗲𝗿-𝗶𝘁𝗲𝗺 𝗺𝗮𝘁𝗿𝗶𝘅)
    const ratingsData = tf.tensor2d(matrix);
    
    // 𝗦𝗽𝗹𝗶𝘁 𝘁𝗵𝗲 𝗱𝗮𝘁𝗮𝘀𝗲𝘁 𝗶𝗻𝘁𝗼 𝘁𝗿𝗮𝗶𝗻𝗶𝗻𝗴 𝗮𝗻𝗱 𝘁𝗲𝘀𝘁 𝘀𝗲𝘁𝘀
    const splitRatio = 0.8;
    const numRatings = ratingsData.shape[0];
    const numTrainRatings = Math.floor(numRatings * splitRatio);
    const trainData = ratingsData.slice([0, 0], [numTrainRatings, ratingsData.shape[1]]);
    const testData = ratingsData.slice([numTrainRatings, 0], [numRatings - numTrainRatings, ratingsData.shape[1]]);

    // 𝗗𝗲𝗳𝗶𝗻𝗲 𝘁𝗵𝗲 𝗦𝗩𝗗 𝗺𝗼𝗱𝗲𝗹
    const svdModel = tf.tidy(() => {
    
        console.log('=================================')
        console.log('Defining the user and item matrix');


        // 𝗗𝗲𝗳𝗶𝗻𝗲 𝘁𝗵𝗲 𝘂𝘀𝗲𝗿 𝗮𝗻𝗱 𝗶𝘁𝗲𝗺 𝗺𝗮𝘁𝗿𝗶𝗰𝗲𝘀
        const numLatentFactors=10;
        const userMatrix = tf.variable(tf.randomNormal([trainData.shape[0], numLatentFactors]));
        const itemMatrix = tf.variable(tf.randomNormal([trainData.shape[1], numLatentFactors]));


        console.log('=================================')
        console.log('Defining adam optimizer .........');


        // 𝗗𝗲𝗳𝗶𝗻𝗲 𝘁𝗵𝗲 𝗼𝗽𝘁𝗶𝗺𝗶𝘇𝗲𝗿
        const learningRate = 0.01;
        const optimizer = tf.train.adam(learningRate);

        console.log('=================================')
        console.log('start training the model ........');

        // 𝗧𝗿𝗮𝗶𝗻 𝘁𝗵𝗲 𝗺𝗼𝗱𝗲𝗹
        for (let i = 0; i <= 275 ; i++) {

            console.log('=================================')
            console.log('start round : ' , i );

            optimizer.minimize(() => {

                // 𝗗𝗲𝗳𝗶𝗻𝗲 𝘁𝗵𝗲 𝗽𝗿𝗲𝗱𝗶𝗰𝘁𝗲𝗱 𝗿𝗮𝘁𝗶𝗻𝗴𝘀
                const predictedRatings = tf.matMul(userMatrix, itemMatrix, false, true);

                const lambda =0.1 ; // regularization strength
                const regularization = tf.add(tf.norm(userMatrix, 'euclidean'), tf.norm(itemMatrix, 'euclidean'));

                // 𝗗𝗲𝗳𝗶𝗻𝗲 𝘁𝗵𝗲 𝗹𝗼𝘀𝘀 𝗳𝘂𝗻𝗰𝘁𝗶𝗼𝗻 (𝗥𝗼𝗼𝘁 𝗠𝗲𝗮𝗻 𝗦𝗾𝘂𝗮𝗿𝗲𝗱 𝗘𝗿𝗿𝗼𝗿)
                const rmse = tf.sqrt(tf.losses.meanSquaredError(predictedRatings ,trainData))
                const loss = tf.add(rmse, tf.mul(lambda, regularization));

                console.log('loss' , loss.arraySync())

                return loss;

            });

            // 𝗘𝘃𝗮𝗹𝘂𝗮𝘁𝗲 𝘁𝗵𝗲 𝗺𝗼𝗱𝗲𝗹 𝗼𝗻 𝘁𝗵𝗲 𝘁𝗲𝘀𝘁 𝘀𝗲𝘁 𝗲𝘃𝗲𝗿𝘆 𝟭𝟬 𝗲𝗽𝗼𝗰𝗵𝘀
            if (i % 5 === 0) {

                const predictedTestRatings = tf.matMul(userMatrix.slice([0, 0], [testData.shape[0] , numLatentFactors] ) , itemMatrix.slice([0, 0], [testData.shape[1] , numLatentFactors] ) , false, true) ;

                const testRmse = tf.sqrt(tf.losses.meanSquaredError(predictedTestRatings ,testData));

                console.log(`Epoch ${i}: Test RMSE = ${testRmse.arraySync()}`);
            }
            
        }

        // 𝗥𝗲𝘁𝘂𝗿𝗻 𝘁𝗵𝗲 𝘁𝗿𝗮𝗶𝗻𝗲𝗱 𝘂𝘀𝗲𝗿 𝗮𝗻𝗱 𝗶𝘁𝗲𝗺 𝗺𝗮𝘁𝗿𝗶𝗰𝗲𝘀
        return [userMatrix, itemMatrix];
    
    });


    const userMatrixData = svdModel[0].arraySync();
    const itemMatrixData = svdModel[1].arraySync();
    const savedModel = { userMatrixData, itemMatrixData };
    fs.writeFileSync('model.json', JSON.stringify(savedModel),(err) => {
        if (err) throw err;
        console.log('Model written to file');
    });

}
