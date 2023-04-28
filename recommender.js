const tf = require('@tensorflow/tfjs');

exports.recommender=(itemId,numRecommendations,ratingsData,isRated,userRatingsArray,itemMatrix)=>{

    if(isRated){

        //* 𝗶𝗳 𝘁𝗵𝗲 𝘂𝘀𝗲𝗿 𝗰𝗵𝗮𝗻𝗴𝗲𝗱 𝗵𝗶𝘀 𝗿𝗮𝘁𝗶𝗻𝗴𝘀 𝗮𝗿𝗿𝗮𝘆 𝗼𝗿 𝗻𝗼𝘁 𝗶 𝗮𝗺 𝗴𝗼𝗻𝗻𝗮 𝗿𝗲-𝗰𝗮𝗹𝗰𝘂𝗹𝗮𝘁𝗲 𝘁𝗵𝗲 𝘂𝘀𝗲𝗿 𝗹𝗮𝘁𝗲𝗻𝘁 𝗳𝗮𝗰𝘁𝗼𝗿 𝘁𝗼 𝘂𝘀𝗲 𝘁𝗵𝗲𝗺 𝘁𝗼 𝗽𝗿𝗲𝗱𝗶𝗰𝘁 𝗵𝗶𝘀 𝗿𝗮𝘁𝗶𝗻𝗴𝘀

        const userRatings=tf.tensor2d([userRatingsArray]);
        const userCenteredRatings = userRatings.sub(tf.mean(ratingsData));
        const itemMatrixSubset = itemMatrix.slice([0, 0], [ratingsData.shape[1] , itemMatrix.shape[1]]);
        const userLatentFactors = tf.matMul(userCenteredRatings, itemMatrixSubset, false, false);
        const predictedUserRatings = tf.matMul(userLatentFactors, itemMatrixSubset, false, true)
        .add(tf.mean(ratingsData))
        .arraySync()[0];

        const topRecommendations = predictedUserRatings
        .map((rating, index) => ({ index, rating }))
        .sort((a, b) => b.rating - a.rating)
        .slice(0, numRecommendations)
        .map(({index})=>{
            return itemId[index];
        });

        //*********************************************************/

        //* 𝘁𝗼 𝗲𝘃𝗮𝗹𝘂𝗮𝘁𝗲 𝘁𝗵𝗲 𝗺𝗼𝗱𝗲𝗹 , 𝗖𝗮𝗹𝗰𝘂𝗹𝗮𝘁𝗲 𝘁𝗵𝗲 𝗥𝗠𝗦𝗘

        const rmse = tf.sqrt(tf.losses.meanSquaredError(predictedUserRatings , userRatingsArray)).arraySync();
        console.log('RMSE',rmse);

        //*********************************************************/

        console.log('Top Recommendation for existed user: ' , topRecommendations);

        return topRecommendations;

    }else{

        //* 𝗿𝗲𝗰𝗼𝗺𝗺𝗲𝗻𝗱𝗶𝗻𝗴 𝘁𝗼𝗽 𝗿𝗮𝘁𝗲𝗱 𝗽𝗿𝗼𝗱𝘂𝗰𝘁𝘀 𝗯𝘆 𝗮𝗹𝗹 𝘂𝘀𝗲𝗿𝘀

        const meanItemsRatings=ratingsData.transpose().arraySync().map(oneItemRatings=>{
            let meanOneItemRatings=0;
            for (let i=0;i<oneItemRatings.length;i++){
                meanOneItemRatings+=oneItemRatings[i];
            }
            return meanOneItemRatings/oneItemRatings.length;
        })

        const topRecommendations=meanItemsRatings
        .map((rating,index)=>({index,rating}))
        .sort((a, b) => b.rating - a.rating)
        .slice(0,numRecommendations)
        .map(({index})=>{
            return itemId[index];
        })

        //*********************************************************/

        console.log('Top Recommendation for new user: ' ,topRecommendations)

        return topRecommendations;

    }

}






