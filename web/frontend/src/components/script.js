document.addEventListener('DOMContentLoaded', function() {
    stopservice();
});
function stopservice() {
    const now = new Date();
    const h = now.getHours();

    if(h >= 21 && h < 9){
        document.body.innerHTML = 
        `<div style="display: flex; justify-content: center; align-items: center; height: 100vh; flex-direction: column; background: black; color: white; text-align: center;">
            <header> 系統維護中
                <p>系統維護時間: 01:00(A.M.) - 05:59(A.M.)
                    <p>目前時間為系統正在維護中，造成您的不便，敬請見諒</p>
                </p>
            </header>
        </div>`
        ;
    }
}