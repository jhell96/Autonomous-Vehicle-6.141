$(document).ready(function(){
	console.log("working");

	//initialize components
	$(".button-collapse").sideNav();
	$('.collapsible').collapsible();
	$('.slider').slider({interval: 3000});
    $('.materialboxed').materialbox();

    // init Masonry
	var $grid = $('.grid').masonry({
	  itemSelector: '.grid-item',
	  percentPosition: true,
	  columnWidth: '.grid-sizer'
	});
	// layout Isotope after each image loads
	$grid.imagesLoaded().progress( function() {
	  $grid.masonry();
	});  

	//simply add the image url into the list of images to cache
	var galleryImageArray = ["images/gallery/gallery1.jpg", "images/gallery/gallery2.jpg", "images/gallery/gallery3.jpg",
						"images/gallery/gallery4.jpg", "images/gallery/gallery5.jpg", "images/gallery/gallery6.png",
						"images/gallery/gallery7.png", "images/gallery/gallery8.jpg", "images/gallery/gallery9.jpg",
						"images/gallery/gallery10.jpg", "images/gallery/gallery11.jpg"]

	var lab2ImageArray = ["images/lab2/gazebo.png", "images/lab2/LIDAR.png", "images/lab2/rviz_lidar_1.png",
						"images/lab2/rviz_lidar_2.png", "images/lab2/zed_color.png"]

	var lab3ImageArray = ["images/lab3/ransac.png", "images/lab3/walking.png"]

	var lab4ImageArray = ["images/lab4/150.png", "images/lab4/150_filtered.png", "images/lab4/ackermann.png",
                        "images/lab4/angle1.png", "images/lab4/blob_detection.png", "images/lab4/chess.png",
                        "images/lab4/chess_highlighted.png", "images/lab4/chess_transformed.png", "images/lab4/contour.png",
                        "images/lab4/goal.png", "images/lab4/grid_overlay.png", "images/lab4/list.txt", "images/lab4/mask.png",
                        "images/lab4/sift_keypoints_3.jpg"]

    var lab5ImageArray = ["images/lab5/belief.png", "images/lab5/condprob.png", "images/lab5/list.txt", "images/lab5/motionmodel1.png", 
                        "images/lab5/motionmodel2.png", "images/lab5/motionmodel3.png", "images/lab5/precompute.png", 
                        "images/lab5/precomputed_sensor_model.png", "images/lab5/tunnelmap.png", "images/lab5/update.png" ]
                        
    var lab6ImageArray = ['images/lab6/dilatedmap.png', 'images/lab6/math1.png', 'images/lab6/math2.png', 'images/lab6/math3.png',
                        'images/lab6/purepursuit.png', 'images/lab6/rrtfull.png', 'images/lab6/rrtnode.png', 'images/lab6/rrtpath.png', 
                        'images/lab6/undilatedmap.jpg']

	preloadImages(galleryImageArray, true);
	preloadImages(lab2ImageArray, true);
	preloadImages(lab3ImageArray, true);
	preloadImages(lab4ImageArray, true);
    preloadImages(lab5ImageArray, true);
    preloadImages(lab6ImageArray, true);

});




//this function loads images in the background into the cache of a browser, so it will load faster
//the second time the page is visited. It caches all the images from all the pages so it loads nicely.
//http://stackoverflow.com/questions/10240110/how-do-you-cache-an-image-in-javascript
function preloadImages(array, waitForOtherResources, timeout) {
    var loaded = false, list = preloadImages.list, imgs = array.slice(0), t = timeout || 15*1000, timer;
    if (!preloadImages.list) {
        preloadImages.list = [];
    }
    if (!waitForOtherResources || document.readyState === 'complete') {
        loadNow();
    } else {
        window.addEventListener("load", function() {
            clearTimeout(timer);
            loadNow();
        });
        // in case window.addEventListener doesn't get called (sometimes some resource gets stuck)
        // then preload the images anyway after some timeout time
        timer = setTimeout(loadNow, t);
    }

    function loadNow() {
        if (!loaded) {
            loaded = true;
            for (var i = 0; i < imgs.length; i++) {
                var img = new Image();
                img.onload = img.onerror = img.onabort = function() {
                    var index = list.indexOf(this);
                    if (index !== -1) {
                        // remove image from the array once it's loaded
                        // for memory consumption reasons
                        list.splice(index, 1);
                    }
                }
                list.push(img);
                img.src = imgs[i];
            }
        }
    }
}
