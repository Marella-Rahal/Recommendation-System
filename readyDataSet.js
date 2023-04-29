const fs=require('fs');

exports.readyDataSet=()=>{

    const csv = fs.readFileSync('ratings.csv','utf-8');
    const rows = csv.split('\n');
    const cols = rows.map(row => row.split(','));
    const matrix = cols.map(row => row.map(val => Number(val)));

    // Users Number
    const userId=new Set()
    matrix.forEach(m=>userId.add(m[0]))

    // Items Number and Items Index 
    var itemId=new Set();
    matrix.forEach(m=>itemId.add(m[1]))
    itemId=Array.from(itemId).sort((a,b)=>a-b);

    // prepare the ratingsData array with zeroes
    var array=[];
    for(let i=0;i<userId.size;i++){
        array[i]=[];
        for(let j=0;j<itemId.length;j++){
            array[i][j]=0;
        }
    }

    // prepare the ratingsData array with the actual values
    for(let i=0;i<matrix.length;i++){
        const row=matrix[i][0]-1;
        const col=itemId.indexOf(matrix[i][1])
        array[row][col]=matrix[i][2]
    }

    array=array.slice(0, Math.ceil(array.length/2))
    array=array.map(row=>row.slice( 0 , Math.ceil(row.length/2) ))

    fs.writeFileSync('readyDataSet.csv',array.map(row => row.join(',')).join('\n'),(err) => {
        if (err) throw err;
        console.log('Matrix dataSet written to file');
    })

}