const fs=require('fs');

exports.dataSet=()=>{

    const array=[];
    const values=[0,1,1.5,2,2.5,3,3.5,4,4.5,5]

    for(let i=0;i<10000;i++){
        array[i]=[];
        for(let j=0;j<2000;j++){
            array[i][j]=values[Math.floor(Math.random() * values.length)];
        }
    }

    fs.writeFileSync('dataSet.csv',array.map(row => row.join(',')).join('\n'),(err) => {
        if (err) throw err;
        console.log('Matrix dataSet written to file');
    })

}