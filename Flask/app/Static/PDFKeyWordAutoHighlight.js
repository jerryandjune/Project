$(document).ready(function () {
    //要执行的js代码段
    ShowFile()
});


function ShowFile() {
    $.ajax({
        url: "/PDFKeyWordAutoHighlight/GetPdfFilePath",
        type: "Post",
        dataType: 'json',
        success: function (data) {
            console.log(data)
            if (data.result.path != null && data.result.path != '') {
                document.getElementById("Preview").src = data.result.path
            }
        },
        error: function (e) {
            $.message({
                message: '处理失败！',
                type: 'error'
            });
        }
    })

}

function ProcessPdf() {
    $.message({
        message: 'Pdf文件转换Html处理中，请稍等！',
        type: 'success'
    });
    //转html
    $.ajax({
        url: "/PDFKeyWordAutoHighlight/ConvertPdf2Html",
        type: "Post",
        dataType: 'json',
        success: function (data) {
            if (data.result.data) {
                document.getElementById("Preview").src = data.result.html
                $.message({
                    message: '转换成功，正在提取关键字自动高亮！',
                    type: 'success'
                });

                $.ajax({
                    url: "/PDFKeyWordAutoHighlight/ProcessPdf",
                    data:
                    {
                        KeyWord: $("#KeyWord").val(),
                    },
                    type: "Post",
                    dataType: 'json',
                    success: function (data) {
                        console.log(data)
                        if (data.result.ret) {
                            document.getElementById("Preview").src = data.result.path
                            //iframe强制刷新
                            $('#Preview').attr('src', $('#Preview').attr('src'))

                            $.message({
                                message: '高亮完成！',
                                type: 'success'
                            });
                        }
                    },
                    error: function (e) {
                        $.message({
                            message: '处理失败！',
                            type: 'error'
                        });
                    }
                })
            }
        },
        error: function (e) {
            $.message({
                message: '处理失败！',
                type: 'error'
            });
        }
    })
}

function Download() {

    $.ajax({
        url: "/PDFKeyWordAutoHighlight/GetHighLightFilePath",
        type: "Post",
        dataType: 'json',
        success: function (data) {
            console.log(data)
            if (data.result.path != '' && data.result.path != null) {
                window.open(data.result.path);
            }
            else {
                $.message({
                    message: '请先上传PDF，然后进行处理！',
                    type: 'error'
                });
            }
        },
        error: function (e) {
            $.message({
                message: '处理失败！',
                type: 'error'
            });
        }
    })

}

function ViewPDF() {
    $.ajax({
        url: "/PDFKeyWordAutoHighlight/GetPdfFilePath",
        type: "Post",
        dataType: 'json',
        success: function (data) {
            console.log(data)
            if (data.result.path != '' && data.result.path != null) {
                document.getElementById("Preview").src = data.result.path
                //iframe强制刷新
                $('#Preview').attr('src', $('#Preview').attr('src'))
            }
            else {
                $.message({
                    message: '请先上传PDF，然后进行处理！',
                    type: 'error'
                });
            }
        },
        error: function (e) {
            $.message({
                message: '处理失败！',
                type: 'error'
            });
        }
    })
}

function ViewHtml() {
    $.ajax({
        url: "/PDFKeyWordAutoHighlight/GetHtmlOrgFilePath",
        type: "Post",
        dataType: 'json',
        success: function (data) {
            console.log(data)
            if (data.result.path != '' && data.result.path != null) {
                document.getElementById("Preview").src = data.result.path
                //iframe强制刷新
                $('#Preview').attr('src', $('#Preview').attr('src'))
            }
            else {
                $.message({
                    message: '请先上传PDF，然后进行处理！',
                    type: 'error'
                });
            }
        },
        error: function (e) {
            $.message({
                message: '处理失败！',
                type: 'error'
            });
        }
    })
}